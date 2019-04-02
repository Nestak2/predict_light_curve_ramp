
# coding: utf-8

# Script for processing the Spitzer data, allowing the user to make lightcurves out of it.
# 
# Running for all files should take around 80 min

# In[1]:


import sys,os
from astropy.io import fits # runs with version version 2.0.8
import pandas as pd # runs with version 0.23.4
import numpy as np # runs with version 1.14.3
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture, aperture_photometry # runs with version 0.4.1
import gc
import time
from datetime import datetime
import copy as cp


# Define the global variables, change them if desired:

# In[ ]:


#####
output_filename = "output_file.txt" # here you determine the file name (and path) where the output data should be saved to
count_factor = 0.32*3.8/0.2021 # adopted from the Agol 2010 paper, they use it there as a multiplication factor, so do I for being consistent with them
binning = True # True or False . Choose if to bin the data points into an averaged data point
frames_per_file = 64 # the number of frames in a single fits-file. we use this to generate the sequence of frames to loop through
time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M') # making a time stamp, as a notation for the name of the directory, where the output files will be saved
input_dir_path = "/Users/narsenov/Documents/Spitzer data" #directory where the input data is stored
output_dir_path = "/Users/narsenov/Documents/synced_docs/UCL internship/code/data_processing_directories/%s_many_parameters_from_IRAF_star%s_bin=%s/" %(time_stamp, star, binning) # define the directory where the outptfiles will be saved
star = 2 # 1 is for the main star, 2 for the secondary star
print_statements_file = "print_statements_file.txt" # file, in which all printed statements from the script will be saved into
#####


# In[ ]:


# creates the directory to save the output fies into
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

# # 3 lines below cause print statements to go into a file. If not desired, outcommand them
# oldStdout = sys.stdout
# logfile = open(output_dir_path + print_statements_file, 'w')
# sys.stdout = logfile

# saving a copy of the file into the target directory,
# but it will only work if executed from a .py-file, not from a jupyter notebook (.ipynb)
file_name = sys.argv[0]
from shutil import copy
copy(file_name, output_dir_path)


# In[ ]:


def fits_files_list(input_dir_path):
    """Do not change the original file names! They need to contain the original notation for this function to work.
    Function that writes all the needed fits files names into one long list,
    after collecting all the desired files from the directory input_dir_path"""
    
    fits_files_list = []
    for path, subdirs, files in os.walk(input_dir_path):
        for name in files:
            # we need only files from chanel 4, containing the "bcd" in the filename and not being a log-file
            # so we extract correspondingly
            if "bcd.fits" in name and "ch4" in path and "log" not in name:
                fits_files_list.append(os.path.join(path, name))
    print("initial file list =", len(fits_files_list))
    return fits_files_list


# In[ ]:


def sort_files_by_time(files_list):
    """Function that opens the fits-files and sorts them corresponding to their observation time.
    Stores and returns the result as a pandas dataframe"""
    time_filename_tupels = [[get_time_from_fits(file_name), file_name] for file_name in files_list]
    df_time_filename_tupels_sorted = pd.DataFrame(time_filename_tupels, columns = ["file_time_UTCS_OBS","fits_file_path"]).sort_values("file_time_UTCS_OBS") 
    print("sorted_files_by_time =", len(df_time_filename_tupels_sorted))
    return df_time_filename_tupels_sorted 
    
def get_time_from_fits(fits_file):
    """Subfunction for the function 'sort_files_by_time', that reads the time (UTCS_OBS) from a single fits file"""
    # the .copy() and gc.collect() commands here are made to prevent "Errorno: [Errno 24] Too many open"
    # this error happens sometime, when to many fits files are opened to quickly and not closed fast enough
    # Therefore the garbage collector gc has to be used here to clear memory
    hdul = fits.open(fits_file)
    time = cp.copy(hdul[0].header["UTCS_OBS"])#.copy()
    hdul.close()
    gc.collect()
    return time


# In[ ]:


def xy_centroid_frame_astropy_extended_output(image_frame, star_index, sharplo, sharphi, roundlo, roundhi):
    """Calculating the x,y position of the main star centroid using the astropy function DAOStarFinder.
    satr_index = 1 or 2 discriminates between evaluation for the main star (1) or the companion (2)"""
    mean, median, std = sigma_clipped_stats(image_frame, sigma=3.0, iters=5)
    
    try:
        daofind = DAOStarFinder(fwhm=3.0, threshold=5*std, sharplo=sharplo, sharphi=sharphi, roundlo=roundlo, roundhi=roundhi) #, sharplo=0.4, sharphi=0.9 #roundlo=-0.15, roundhi=0.15
        sources_dao = daofind(image_frame - median)
        sources_dao.sort("flux")
        df_star = pd.DataFrame(np.array(sources_dao[[-star_index]]))# the slicing selects here between the user choosen main or secondary star 
        df_star["num_sources"] = len(sources_dao)
    # the exception below is needed in the cases, when no source is being found by dao_find in the frame, which happens
    # or the the frame is erroros and contains no data
    # In these cases the exception creates a dummy data row, containing everywhere values=0.1 as flags
    except:
        columns = ["id", "xcentroid","ycentroid","sharpness","roundness1","roundness2","npix","sky", "peak","flux","mag","num_sources"]
        #columns = ["xcentroid","ycentroid","sharpness","roundness1","num_sources"]
        data = np.ones((1,len(columns)))/10.
        df_star = pd.DataFrame(data, columns=columns)
    return df_star


# In[ ]:


def get_flux_by_astropy(image_frame,x,y,centroid_rad):
    """Calculating the object flux using the photutils and from there the
    CircularAperture function. See here: http://photutils.readthedocs.io/en/stable/aperture.html
    Note that the flux is NOT being backgroung corrected!"""
    apertures = CircularAperture([x,y], r=centroid_rad)
    phot_table = aperture_photometry(image_frame, apertures)
    return phot_table["aperture_sum"][0] 


# In[ ]:


def return_im_data(fits_file):
    """Function that extracts the image data from a fits-file.
    The original fits-files contain the images as cubes with dimensions 32*32*64,
    32*32 being the spatial x*y pixel_dimensions and 64 being the number of individual frames of the image"""
    # the .copy() and gc.collect() commands here are made to prevent "Errorno: [Errno 24] Too many open"
    # this error happens sometime, when to many fits files are opened to quickly and not closed fast enough
    # Therefore the garbage collector gc has to be used here to clear memory
    hdul = fits.open(fits_file)
    image_data = hdul[0].data.copy()
    hdul.close()
    gc.collect()
    return image_data


# In[ ]:


def binning_func(df_unavereged, frames_to_average_over):
    """The function takes in the star-data (x,y,sharpnes,roundnes....,flux),
    than bins the data into bins of the chosen lenght of frames_to_average_over and also creates the average of the bin.
    frames_to_average_over has to be 64, if consistency with the original paper of Agol 2010 is desired,
    but can be any integer divisor of the lenghth of the dataframe (number of data points)"""
    # extract the data as a numpy array, instead of a dataframe, because easier to handle in binning and averaging with numpy functions
    data_unav = df_unavereged.values
    num_columns = len(df_unavereged.columns)
    data_av = np.mean(data_unav.reshape(-1,frames_to_average_over,num_columns), axis=1)
    df_averaged = pd.DataFrame(data_av, columns=df_unavereged.columns)
    return df_averaged


# In[ ]:


def main_star1(file_with_timesorted_fits_files=None, 
                star_index=1, centroid_rad=4.5, 
               binning=False, frames_to_average_over=64, # if binning==False then frames_for_average is being ignored
               lower_flux_boundary=63500, upper_flux_boundary=71000, 
               sharplo=0.1, sharphi=0.9, 
               roundlo=-0.9, roundhi=0.9): # the natural spread of the data points, for star 1, is between sharplo=0.6, sharphi=0.85, roundlo=-0.5, roundhi=0.65
    """Main function for star1, calls all others consequently to create the light curve.
    Identical to main_star2, beside the different values for the keywords"""    
    
    # keeps track of when the main function started, displays eventually how long it takes to run
    start_time = time.time()
    
    # if file_with_timesorted_fits_files == None, then creates a list of time-sorted fits-file names (ca. 40min)
    # else, if such a list already exists, then you can provide it's file location
    # as a keyword in the form file_with_timesorted_fits_files="path/to/file"
    if file_with_timesorted_fits_files == None:
        print(1)
        initial_file_list = fits_files_list(input_dir_path)
        print(2)
        df_time_sorted_files = sort_files_by_time(initial_file_list)
        print(3)
#         df_time_sorted_files.to_csv(output_dir_path+output_filename, encoding='ascii',float_format='%10.5f', mode='w', header=True, index=False)
        #df_time_files_sort = function_save_to_txt(time_sorted_files_tupels, ["file_time_UTCS_OBS","fits_file_path"], output_dir_path+"time_vs_sorted-fits-files-test%s.txt" %(time_stamp))
        files_list = df_time_sorted_files["fits_file_path"]
    else:
        time_sorted_files_tupels = pd.read_csv(file_with_timesorted_fits_files, header=0)[6023/64:6023/64+2]#[0*690:0*690+2]
        files_list = time_sorted_files_tupels["fits_file_path"]
    
#     files_list = files_list[:10]
    
    # perform the main computation of centroid coordinates and integrated flux:
    print("loop exited")
    # create a list of file names, each with n=frames_per_file duplicates,
    # since there are n frames in each fits-file, that we extract. In out case n=64
    f_list_nested = [[f_name]*frames_per_file for f_name in files_list]
    f_list = [item for sublist in f_list_nested for item in sublist]
    
    # put all frames in one long array
    frames_list = [frame for fits_file in files_list for frame in return_im_data(fits_file)]
    
    # get centroid locations.... and more features 
    features_table = pd.concat([xy_centroid_frame_astropy_extended_output(frame, star_index, sharplo, sharphi, roundlo, roundhi) for frame in frames_list])
    # go through all the frames and extract the integrated flux
    flux_integrated = np.asarray([get_flux_by_astropy(frame,x,y,centroid_rad)*count_factor for frame,x,y in zip(frames_list,features_table.xcentroid,features_table.ycentroid)])
    # create a new column and insert the integrated fluc there
    
    ##### in the next 10 lines the outliers are being taken out and replaced with previous values in the dataframe
    features_table['flux_integrated'] = flux_integrated
    # - include a column called "outlier_flag". Set it to 0, except where NaN, set it there to 1
    features_table['outlier_flag'] = np.zeros((len(features_table),1))
    # - go through the big pandas dataframe and replace the outliers with previous NaN
    features_table.loc[(features_table.flux_integrated > upper_flux_boundary) | (features_table.flux_integrated < lower_flux_boundary), ['flux_integrated', 'xcentroid', 'ycentroid', 'outlier_flag']] = np.nan, np.nan, np.nan, 1
    # - substitute all the NaNs with previous value
    features_table.fillna(method='ffill', inplace=True)
    # reset the id column of the dataframe to have continous numbering (0,1,2,3...), because until here it has numbering (0,0,0...)
    # features_table.reset_index(drop=True, inplace=True)
    #####
    
    # if set to ==True will bin the data in the chosen bin size =frames_to_average_over
    # Note, frames_to_average_over needs to be an integer divisor of the number of data points (lenght of the dataframe) to work
    if binning == True:
        features_table = binning_func(features_table, frames_to_average_over)
    
    # save the dataframe to a txt-file
    features_table.to_csv(output_dir_path+output_filename, encoding='ascii',float_format='%10.5f', mode='w', header=True, index=False)
    
    # print eventually the time needed for the execution of the main-function
    print("--- %s seconds ---" % (time.time() - start_time))
    # line below serves for saving print statements to file
    return features_table


# In[ ]:


def main_star2(file_with_timesorted_fits_files=None, 
                star_index=2, centroid_rad=1.5, 
               binning=False, frames_to_average_over=64, # if binning==False then frames_for_average is being ignored
               lower_flux_boundary=1500, upper_flux_boundary=2500, 
               sharplo=0.1, sharphi=0.9, 
               roundlo=-0.9, roundhi=0.9): # the natural spread of the data points, for star 1, is between sharplo=0.6, sharphi=0.85, roundlo=-0.5, roundhi=0.65
    """Main function for star2, calls all others consequently to create the light curve.
    Identical to main_star1, beside the different values for the keywords."""    
    
    # keeps track of when the main function started, displays eventually how long it takes to run
    start_time = time.time()
    
    # if file_with_timesorted_fits_files == None, then creates a list of time-sorted fits-file names (ca. 40min)
    # else, if such a list already exists, then you can provide it's file location
    # as a keyword in the form file_with_timesorted_fits_files="path/to/file"
    if file_with_timesorted_fits_files == None:
        print(1)
        initial_file_list = fits_files_list(input_dir_path)
        print(2)
        df_time_sorted_files = sort_files_by_time(initial_file_list)
        print(3)
#         df_time_sorted_files.to_csv(output_dir_path+output_filename, encoding='ascii',float_format='%10.5f', mode='w', header=True, index=False)
        #df_time_files_sort = function_save_to_txt(time_sorted_files_tupels, ["file_time_UTCS_OBS","fits_file_path"], output_dir_path+"time_vs_sorted-fits-files-test%s.txt" %(time_stamp))
        files_list = df_time_sorted_files["fits_file_path"]
    else:
        time_sorted_files_tupels = pd.read_csv(file_with_timesorted_fits_files, header=0)[6023/64:6023/64+2]#[0*690:0*690+2]
        files_list = time_sorted_files_tupels["fits_file_path"]
    
#     files_list = files_list[:]
    
    # perform the main computation of centroid coordinates and integrated flux:
    print("loop exited")
    # create a list of file names, each with n=frames_per_file duplicates,
    # since there are n frames in each fits-file, that we extract. In out case n=64
    f_list_nested = [[f_name]*frames_per_file for f_name in files_list]
    f_list = [item for sublist in f_list_nested for item in sublist]
    
    # put all frames in one long array
    frames_list = [frame for fits_file in files_list for frame in return_im_data(fits_file)]
    
    # get centroid locations.... and more features 
    features_table = pd.concat([xy_centroid_frame_astropy_extended_output(frame, star_index, sharplo, sharphi, roundlo, roundhi) for frame in frames_list])
    # go through all the frames and extract the integrated flux
    flux_integrated = np.asarray([get_flux_by_astropy(frame,x,y,centroid_rad)*count_factor for frame,x,y in zip(frames_list,features_table.xcentroid,features_table.ycentroid)])
    # create a new column and insert the integrated fluc there
    
    ##### in the next 10 lines the outliers are being taken out and replaced with previous values in the dataframe
    features_table['flux_integrated'] = flux_integrated
    # - include a column called "outlier_flag". Set it to 0, except where NaN, set it there to 1
    features_table['outlier_flag'] = np.zeros((len(features_table),1))
    # - go through the big pandas dataframe and replace the outliers with previous NaN
    features_table.loc[(features_table.flux_integrated > upper_flux_boundary) | (features_table.flux_integrated < lower_flux_boundary), ['flux_integrated', 'xcentroid', 'ycentroid', 'outlier_flag']] = np.nan, np.nan, np.nan, 1
    # - substitute all the NaNs with previous value
    features_table.fillna(method='ffill', inplace=True)
    # reset the id column of the dataframe to have continous numbering (0,1,2,3...), because until here it has numbering (0,0,0...)
    # features_table.reset_index(drop=True, inplace=True)
    #####
    
    # if set to ==True will bin the data in the chosen bin size =frames_to_average_over
    # Note, frames_to_average_over needs to be an integer divisor of the number of data points (lenght of the dataframe) to work
    if binning == True:
        features_table = binning_func(features_table, frames_to_average_over)
    
    # save the dataframe to a txt-file
    features_table.to_csv(output_dir_path+output_filename, encoding='ascii',float_format='%10.5f', mode='w', header=True, index=False)
    
    # print eventually the time needed for the execution of the main-function
    print("--- %s seconds ---" % (time.time() - start_time))
    # line below serves for saving print statements to file
    return features_table


# In[ ]:


# execute the main function here
if star == 1:
    df_flux_calculator_astropy_accurate_mean_images = main_star1(binning=binning) #, file_sorted_time="time_vs_sorted-fits-files20June.txt"
if star == 2:
    df_flux_calculator_astropy_accurate_mean_images = main_star2(binning=binning)
    
    
# sys.stdout = oldStdout
# logfile.close()

