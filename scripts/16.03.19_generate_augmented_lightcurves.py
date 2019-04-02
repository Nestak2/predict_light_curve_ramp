
# coding: utf-8

# The aim ot the script here is to use a real light curve, bin it in different ways and to create so augmented light curves that will serve for training the neural network on them to produce augmented ones.
# 
# Steps:
# 1. Read in the file with the unbinned fluxes with pandas
# 2. Bin the data like before in sequences of 64
# 3. Out of each sequence take out 15%,20%,25% or just n% of the data points
# 4. Perform again the average of the flux in each of these bins of sequences
# 5. Do this 100 times and save into a txt file with pandas
# 
# Later do a masking of the points

# In[14]:


import sys,os
#from astropy.io import fits
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
from numpy import unravel_index
#from astropy.stats import sigma_clipped_stats
#from photutils import DAOStarFinder
import gc
#from photutils import CircularAperture
#from photutils import aperture_photometry


# Define the global variables, change them if desired:

# In[15]:


# how many augmented lightcurves should be included in the output file
num_lightcurves = 3
# fraction of the data points to be used for averaging
fraction = 0.5

# decide how the output data to be created. If option=1 the output data will be binned to the average of a fraction of the datapoints
# If option=2 then only noise will be added to the data
option=2

# the size of the bins for averaging
n_frames = 64

# the number of data points/vectors building a single lightcurve
len_lc = 690*64

# the samples don't need to be too toroughly randomized. with fraction=0.5 and n_frames=64 there are about 64!/((64-10)!*10!)~10**18 possible configurations of choosing
# 0.5=50% of the frames in a sequence 

# file_path of the input lightcurve used to create augmented lightcurves out of it by different binning
file_path_in = '/Users/narsenov/Documents/synced_docs/UCL internship/code/xycentr_flux_file#_frame#_file26June-singlepx-astropyfluxcalc_lenvec=4.txt'
#"/home/nestor/Documents/internship-UCL-2018/code/xycentr_flux_file#_frame#_file26June-singlepx-astropyfluxcalc_lenvec=4.txt"

#the file path to save to
file_path_out = 'test_augm_lc.txt'
# the input data
df_in = pd.read_csv(file_path_in, header=0)
# select only the last light curve in the data
df_in = df_in[-len_lc:]


# In[18]:


def rebin_ramp(fraction, num_lightcurves, option, n_frames, df_in, file_path_out, len_lc):
    """Function that creates augmented data. The function has 2 operation modes.
    In option 1 it takes input data, bins it,
    takes out a random fraction of data points from each bin and calculates the average, creating a lightcurve in shape similar to Agol's.
    This option is suitable, if the user want to bin the input data.
    In option 2 it adds noise within the standard deviation of the data to the data. Option 2 is suitable for already binned data
    or if the user doesn't want to bin the input data.
    Eventually, this is saved into a txt file with pandas, the file is being appended if already exists, instead of overwritten."""
    
    if option==1:
        fluxes_binned = df_in["fluxes_frames"].values.reshape(-1, n_frames)
        x_centr_binned = df_in["x_centr"].values.reshape(-1, n_frames)
        y_centr_binned = df_in["y_centr"].values.reshape(-1, n_frames)
        
        for i in range(num_lightcurves):
            x_centr_frac_mean = np.array([])
            y_centr_frac_mean = np.array([])
            f_centr_frac_mean = np.array([])

            for seq_x, seq_y, seq_f in zip(x_centr_binned, y_centr_binned, fluxes_binned):
                # here we choose randomly fraction*n_frames indeces, to extract the corresponding items from each of the previously formed sequences of 64.
                # means, from all the items in a sequence with indeces (0,1,2,3...62,63) we pick randomly a fixed fraction (8,13,22,25,31...)
                rand_ind = np.random.choice(range(n_frames), int(fraction*n_frames), replace=False)
                # but we need to be consistent and use the same selection of indeces for x,y and flux from the same sequence of 64, because they are correlated being measured at the same time from Spitzer

                seq_x_frac = seq_x[rand_ind]
                x_centr_frac_mean = np.append(x_centr_frac_mean, np.mean(seq_x_frac))

                seq_y_frac = seq_y[rand_ind]
                y_centr_frac_mean = np.append(y_centr_frac_mean, np.mean(seq_y_frac))

                seq_f_frac = seq_f[rand_ind]
                f_centr_frac_mean = np.append(f_centr_frac_mean, np.mean(seq_f_frac))

            data = np.stack([x_centr_frac_mean, y_centr_frac_mean, f_centr_frac_mean], axis=1)
            df = pd.DataFrame(data, columns=df_in.columns[-3:])
            # the line below has the mode "mode='a'" for appending, so it simply appends the file once created
            df.to_csv(file_path_out, encoding='ascii',float_format='%10.5f', mode='a', header=False, index=True)

            if i%100==0:
                print(i)
                
    # if you don't want the output data to be binned, it will just get noise added to it
    if option==2:
        for i in range(num_lightcurves):
            # 1. calculate the standard deviation over a region for the 3 relevant quantities
            std_flux = np.std(df_in["fluxes_frames"][int(0.9*len(df_in)):int(0.95*len(df_in))])
            std_x = np.std(df_in["x_centr"][int(0.9*len(df_in)):int(0.95*len(df_in))])
            std_y = np.std(df_in["y_centr"][int(0.9*len(df_in)):int(0.95*len(df_in))])
            
            # 2. add random noise to this 3 quantities with standard deviation = the one previously calculated
            flux = df_in["fluxes_frames"] + np.random.normal(0,std_flux,len(df_in))
            x = df_in["x_centr"] + np.random.normal(0,std_x,len(df_in))
            y = df_in["y_centr"] + np.random.normal(0,std_y,len(df_in))
            
            # get the data into a dataframe and save it
            data = np.stack([x, y, flux], axis=1)
            df = pd.DataFrame(data, columns=df_in.columns[-3:])
            # the line below has the mode "mode='a'" for appending, so it simply appends the file once created
            df.to_csv(file_path_out, encoding='ascii',float_format='%10.5f', mode='a', header=False, index=True)
            
            if i%100==0:
                print(i)
    return df


# In[24]:


rebin_ramp(fraction=fraction, num_lightcurves=num_lightcurves, option=option, n_frames=n_frames, df_in=df_in, file_path_out=file_path_out, len_lc=len_lc)
# the function appends data to the file_path_out, read in the output file after the function has been executed
df_out = pd.read_csv(file_path_out, header=0)


# Do a plot of the created augmented light curve to see if it has been performed well

# In[27]:


plt.figure(figsize=(20,10))
plt.scatter(range(690*64), np.loadtxt(file_path_out,delimiter=",")[-690*64:,3], s=4, label="averaged flux")
# plt.scatter(range(690), np.mean(df_in["fluxes_frames"][-690*64:].values.reshape(-1, 64), axis=1), s=4, )
plt.grid()

