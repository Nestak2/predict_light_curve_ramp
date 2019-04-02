# predict_ligtht_curve_ramp

Prediction of a star light curve behaviour with a neural network

## What is this code for?
The scripts here provide neural network predictions of the behaviour of the light curve of star HD 189733 as observed by Agol et al. (2010) with the Spitzer Space Telescope. The star experiences an exoplanet transit, but its light curve is contaminated with a ramp in the flux with instrumental origin. The aim of the work is to correct for the artificial ramp in the light curve by using a neural network written in Tensorflow and Keras, that predicts the behaviour of the light curve. The neural network is not fully functional, in the sense that it learns the qualitative behaviour of the ramp, but it doesn’t provide predictions accurate enough to allow correcting it in the light curve.
 
We provide in the repository a script that
-	creates the observed lightcurve out of the original Spitzer fits-files
-	creates augmented light curves for the purpose of training on them
-	plots the prediction of the neural network

We also provide data files created by this scripts that
-	contain, as provided by Agol et al. (2010), 6 lightcurves of the main star, unbinned
-	-||- main star, averaged in bins of 64 data poins
-	-||- secondary star, unbinned
-	-||- secondary star, averaged in bins of 64 data poins
-	contain augmented data for training, in the form of 900 lightcurves created by different binning of the real light curve
-	contain the trained neural network model

Show picture of lightcurve, show some predictions


## Instructions for use:
We provide the data files needed for training the neural network and predicting sequences of the lightcurve, but if desired by the user he can reproduce the data files with the given scripts. The recommended order of work is the following:
- create the real light curve of the star by using the script create-lightcurve.py. The user will need to have the observational files of star HD 189733 used by Agol et al. (2010). The output file of the created light curve is in the files containing “real_lc” in their names. “Star1” or “2” refers to either the flux being from the main star (1) or from the secondary star (2) in the frame. “Bin=True” or “False” refers to the flux being binned or not
- afterwards, create a file containing a series of augmented light curves to be used for training of the neural network with the script “generate_augmented_lightcurves.py”. An output file of this script is provided, containing “augm_lc” in the file name
- afterwards, execute the neural network in the script “neural_network.py”. It will create as an output a file containing the model allowing to make predictions based on this model. An output file of this script is provided, containing “savemodel.h5” in the file name. The script “neural_network.py” will also create a log-file than can show the training performance with tensorboard
- create predictions of the model and plot them as well with the script “plot_lstm_model.py”

## Required modules:
- python system version: 3.6.6
- other modules from the Python Standard Library: os, gc, time, datetime, copy
External modules:
- pandas: version 0.23.4
- numpy: version 1.14.3
- astropy: version 2.0.8
- photutils: 0.4.1
- tensorflow: 1.10.0

### Creating the lightcurve – create-lightcurve.py :
First one has to create the lightcurve of the star, before training and predicting with a neural network on it. This requires the user to have the Spitzer observational fits-file available, with their file names following the original Spitzer notation (the script is created to look for the original file notation). We also provide the extracted lightcurves in this repository, in case the user doesn’t want to extract it on his own.

For creating the light curve the user has to run the file create-lightcurve.py. In the preamble of the file the global variables are provided and can be altered by the user:

output_filename = "output_file.txt" - here you determine the file name (and path) where the output data should be saved to
count_factor = 0.32*3.8/0.2021 - adopted from the Agol et al. 2010 paper, they use it there as a multiplication count factor for the flux, so does the script for being consistent with them
frames_per_file = 64 - the number of frames in a single fits-file. we use this to generate the sequence of frames to loop through
time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M') - making a time stamp, as a notation for the name of the directory, where the output files will be saved
input_dir_path = "/home/nestor/Documents/internship-UCL-2018/test-data" - directory in which the input data is stored
output_dir_path = "./data_processing_directories/%s_many_parameters_from_IRAF_star1/" %(time_stamp) - directory in which the outptfiles will be saved
star = 1 or 2 – the star for which the lightcurve should be obtained, 1 is the main star, 2 is the secondary
print_statements_file = "print_statements_file.txt" - file, in which all printed statements from the script will be saved into

File create-lightcurve.py contains a main function main_star1(), which calls all other functions and creates the lightcurve. The function takes in input parameters as keywords, that the user can alter:

file_with_timesorted_fits_files=None – defaults is set to “None” and in this case the function will create a file of such kind, listing in a time order all fits-files located in input_dir_path containing flux values. When the function is ran a next time the variable file_with_timesorted_fits_files can be set equal to the already created file from the first run, hence preventing the function from sorting again the fits-files by time and saving a significant amount of time

star_index=1- allows the user to choose for which star in the field of view the light curve should be extracted – star_index=1 refers to the main star HD 189733 and star_index=2 refers to the companion star

centroid_rad=4.5 – radius of the star in pixel. Needed by aperture_photometry for determining the integrated flux of the star. Should be asjusted with care, as main star has a companion at a distance of ~6 pixel. The value of 4.5 for the main star is adopted from Agol et al. (2010)
 
binning=False – if True bins the lichtcurve into data points, each being the average over n unbined data points, with n=frames_to_average_over
frames_to_average_over=64 - if binning=False then frames_for_average is being ignored, if binning=True, than frames_to_average_over gives the size of a bin of data-points

lower_flux_boundary=63500, upper_flux_boundary=71000 – the lower and upper limit for the flux of datapoints. Outside this range data points are considered outliers and are substituted with chronologically previous data point. These values have been empirically determined to be suitable for the main star

sharplo=0.1, sharphi=0.9 – lower and upper limit for the range of sharpness of the detected objects, required by DaoStarfinder. Outside this range objects are being ignored as artefacts. Should be adjusted with care by the user

roundlo=-0.9, roundhi=0.9 – same as for sharplo and sharphi, but regarding the roundness of the object

The user should consider that the flux of the light curve is not background corrected

File create-lightcurve.py contains a second main function main_star2(), which when ran provides the light curve of the companion star in the field of view. The function main_star2() is identical to main_star1(), beside the default values of four keywords:
- star_index=2, centroid_rad=1.5
- lower_flux_boundary=1500, upper_flux_boundary=2500
These values have been empirically determined to be suitable for the companion star.  

### Generate augmented lightcurves – generate_augmented_lightcurves.py
The aim of the script here is to use a real light curve, bin it in different ways and to create so augmented light curves that will serve for training the neural network on them to produce augmented ones.
The script contains comments that explain how to use it. Have a look at the preamble where the global variables are defined.

### Neural network – neural_network.py
Code creating the neural network, training on augmented lightcurves and saving a file of the trained model.
The script contains comments that explain how to use it. Have a look at the preamble where the global variables are defined.


### Plot the LSTM model – plot_lstm_model.py
A script that loads the model saved by neural_network.py, does a prediction of a sequence of data with it and plots the prediction.
The script contains comments that explain how to use it. Have a look at the preamble where the global variables are defined.


References:
1.	Agol, Eric, et al. "The climate of HD 189733b from fourteen transits and eclipses measured by Spitzer." The Astrophysical Journal 721.2 (2010): 1861.
