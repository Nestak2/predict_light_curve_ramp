
# coding: utf-8

# ## Main code creating the neural network, training on data and saving a file of the trained model

# In[1]:


import numpy as np
import pandas as pd
import json
import os
import sys
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, LambdaCallback # RemoteMonitor

import time
from datetime import datetime


# In[2]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[3]:


# cd "gdrive/My Drive/Colab Notebooks/UCL_internsh_for_github"


# In[4]:


# !ls


# Initial definitions and global variables. Change if desired

# In[5]:


len_x_sequence = 100 # the number of data-vectors we use for building each input-sequence
len_y_sequence = 100 # the number of data-vectors we use for building each output-sequence
# one should use test data, beside validation data, because the validation data is used at the end of each epoch
# so the neural network will get biased towards it. At the end one shoudl therefore check the model with the test data
# and the validation and test loss should not be far from each for an unbiasly trained model
frac_test_data = 0.95
# choosing the training option, 1 or 2. Option 1, skips all data during the transit, and possibly after the transit, if len_x_sequence
# and len_y_sequence are too large. Option 2 takes also the transit as data, but masks it as 0 (it has not been toroughly tested, if this
# approach works properly)
training_option = 1 #or 2

num_epochs= 10
num_lc= 900
batch_size=64

validation_split=0.3
datapoints_per_lc = 690 # the number of (data points)/(light curve). Should be 690 if the light curve is averaged and 690*64 if not averaged 

# Creating the training set - 
# the ingress, egress values here will have to be changed by a factor 4140/690 or 4140*64/690
center_transit = 455#(np.abs(x_scaled - np.mean(x_scaled[y_scaled < 0.1]))).argmin()
ingress = 120 #110
egress = 120 #125

time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
dir_path = "./simulations_directories/%s_gith_option1_lenseq=%s_epochs=%s_traininglcs=%s_batch_s=%s/" %(time_stamp, len_x_sequence, num_epochs, num_lc, batch_size)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

file_name = sys.argv[0]
from shutil import copy
copy(file_name, dir_path)

start_time = time.time()


# In[6]:


import sklearn.preprocessing


# In[7]:


data_full = np.loadtxt("/Users/narsenov/Documents/synced_docs/UCL internship/code/good working code and files/for_github/data/2019-03-20_11_19_augm_lc.txt", skiprows=0, delimiter=",") #/Users/narsenov/Documents/synced_docs/UCL internship/code/lightcurves_augmented_lenved=4_fromfinalramp_Aug7.txt
data = data_full[:num_lc*datapoints_per_lc]

# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc_data = MinMaxScaler(feature_range = (10**(-10), 1.0))
data_scaled = sc_data.fit_transform(data)

X,y = [],[]

# we bin the long file containing 2000 ramps one after the other to be in bins of each one ramp, with a lenght of the bins of 690 vectors
data_sequenced = np.reshape(data_scaled, (-1, datapoints_per_lc, data_scaled.shape[1]))

if training_option == 1:
    # selecting data for training option 1. - withoug masking the transit, just skipping it
    for individual_ramp in data_sequenced:
        # the indeces below are chosen so that the data before and after transit is selected, but not during
        for i in list(range(len_x_sequence, center_transit - ingress - len_y_sequence)) + list(range(center_transit+egress+len_x_sequence, datapoints_per_lc - len_y_sequence)):
            indeces = range(i-len_x_sequence , i) #[i for j in (range(i-len_x_sequence , i), range(center_transit+egress, len(individual_ramp))) for i in j] #range(i-len_x_sequence , i) + range(center_transit+egress, len(individual_ramp))
            sequence_of_vectors = individual_ramp[indeces,:]
            X.append(sequence_of_vectors)
            y.append(individual_ramp[i:i+len_y_sequence,:]) # the slicing here might be a problem, look it up 
        
if training_option == 2:
    # selecting data for training option 2. - transit data included, but masked later (this option has not been tested!)
    for individual_ramp in data_sequenced:
        # set flux-values in transit to 0
        individual_ramp[center_transit - ingress: center_transit + egress,3:] = 0.0
        for i in range(len_x_sequence, len(individual_ramp) - len_y_sequence):
            sequence_of_x_vectors = individual_ramp[i-len_x_sequence:i,:]
            sequence_of_y_vectors = individual_ramp[i:i+len_y_sequence,:]
            X.append(sequence_of_x_vectors)
            y.append(sequence_of_y_vectors)         
        
X, y = np.array(X), np.array(y)

# split X and y into X_train, y_train and X_test,y_test
X_test, y_test = X[int(frac_test_data * len(X)):], y[int(frac_test_data * len(y)):] 
X_train, y_train = X[:int(frac_test_data * len(X))], y[:int(frac_test_data * len(y))] 


# In[8]:


# Building the model
import tensorflow as tf

model = tf.keras.models.Sequential()

# in line below, the masking of 0s is for the training option 2, where we neglect the transit with this mask
# the line does not ifluence training performed in the way of option 2z
model.add(tf.keras.layers.Masking(mask_value = 0.0, input_shape = (None, X_train.shape[2],)))
#model.add(tf.keras.layers.LSTM(units = 512, activation = 'tanh', return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2]) )) # input_shape = (X_train.shape[1], 1)
#model.add(tf.keras.layers.Dropout(0.20))
#model.add(tf.keras.layers.LSTM(units = 32, activation = 'tanh', return_sequences = True))
#model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.LSTM(units = 16, activation = 'tanh', return_sequences = True)) #, return_sequences = True
model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.LSTM(units = 8, activation = 'tanh', return_sequences = True)) #(X_train.shape[0], X_train.shape[1], X_train.shape[2])
model.add(tf.keras.layers.Dropout(0.20))
model.add(tf.keras.layers.Dense(units = X_train.shape[2])) # try 20, instaed of 80!

# model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
# model.fit(X_train, y_train, epochs = 1000, batch_size = 64) # this is where the long calculation happens #50   


# code from the weather forcast template:

# In[9]:


# validation_data = (np.expand_dims(X_test, axis=0), np.expand_dims(y_test, axis=0))
validation_data = (np.reshape(X_test, (1,-1,X_test.shape[2])), np.reshape(y_test, (1,-1,y_test.shape[2])))


# In[10]:


# the ifclause serves for omitting errors when value range defined in training is exceeded by test/prediction
if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))
    

warmup_steps = 5

# below is a custom made loss function, but the function didn't showed to increase the performance,
# so we work insted with the keras mean squared error function

# def loss_mse_warmup(y_true, y_pred):
#     """
#     Calculate the Mean Squared Error between y_true and y_pred,
#     but ignore the beginning "warmup" part of the sequences.
    
#     y_true is the desired output.
#     y_pred is the model's output.
#     """

#     # The shape of both input tensors are:
#     # [batch_size, sequence_length, num_y_signals],
#     # with [~10**6, ~100, 4]. 

#     # Ignore the "warmup" parts of the sequences by taking slices through sequence length.
#     # Also, take into acount only the features "num"(=proxy for time) and "flux", not the centroid position
    
#     y_true_slice = y_true[:, warmup_steps:, ]  # = y_train
#     y_pred_slice = y_pred[:, warmup_steps:, ]  # = X_train
    
#     # where does the input come from? - From model.fit(X_train, y_train, ...), with y_train=y_true and X_train=y_pred. But it makes no sense!

#     print ("y_true_slice.shape =", y_true_slice.shape)
#     print ("y_pred_slice.shape =", y_pred_slice.shape)
    
#     # These sliced tensors both have this shape:
#     # [batch_size, sequence_length - warmup_steps, num_y_signals]

#     # Calculate the MSE loss for each value in these tensors.
#     # This outputs a 3-rank tensor of the same shape.
#     loss = tf.losses.mean_squared_error(labels=y_true_slice,
#                                         predictions=y_pred_slice)

#     # Keras may reduce this across the first axis (the batch)
#     # but the semantics are unclear, so to be sure we use
#     # the loss across the entire tensor, we reduce it to a
#     # single scalar with the mean function.
#     loss_mean = tf.reduce_mean(loss)

#     return loss_mean


# In[11]:


model.compile(optimizer = RMSprop(lr=1e-3), loss = 'mean_squared_error') # loss_mse_warmup
model.summary()


# In[12]:


# create check points for the performance of the code
path_checkpoint = dir_path + 'checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)

# early stopping - if performace of the NN stagnates or become worse the training will be stopped earlier
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
callback_tensorboard = TensorBoard('./simulations_directories/logs/%s' %(time_stamp), histogram_freq=1, write_graph=True)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1)

## next 15 lines below in the cell serve for saving the training progress into files, so it can be checked how the training performs while running on server
# command below gives errors, so I have outcommented it. It says: "ImportError: RemoteMonitor requires the `requests` library" and I don't know how to include the request properly
# callback_monitor = RemoteMonitor(root='./', path='./callbacks_dir/', field='data', headers=None)#send_as_json=False
callback_csvlogger = CSVLogger(dir_path + 'log_file1.csv', append=True, separator=';')

json_log = open(dir_path + 'log_file2.json', mode='wt', buffering=1)
callback_jsonlogging = LambdaCallback(
    on_batch_end=lambda batch, logs: json_log.write(
        json.dumps({'-batch': batch, 'loss': logs['loss'].tolist()}) + '\n'),
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'EPOCH': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close())

callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr, callback_csvlogger, callback_jsonlogging] #callback_monitor,

#oldStdout = sys.stdout
#file = open(dir_path + 'log_file3.txt', 'w')
#sys.stdout = file


# In[13]:


# model.fit(X_train, y_train, epochs=10, steps_per_epoch=50, validation_data=validation_data) #, validation_data=validation_data, callbacks=callbacks
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks) # validation_data=validation_data,

# line below serves for saving progress to file
#sys.stdout = oldStdout


# In[14]:


# Because we use early-stopping when training the model, it is possible that the model's performance has worsened on the test-set for several epochs
# before training was stopped. We therefore reload the last saved checkpoint, which should have the best performance on the test-set.

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# In[15]:


# evaluate the model with the test data
# the validation and test loss should not be far from each for an unbiasly trained model
result = model.evaluate(x=X_test, y=y_test)
print("loss (test-set):", result)


# In[16]:


# save the model to file, that allows to load/recreate the model without running it again
model.save(dir_path + "savemodel.h5")

print("%s sec" % (time.time() - start_time))


# ## Next steps:
# - +include also training after the transit
# - +create script version with training on the whole light curve with masked transit
# - +check if for the creation of the augmented light curve noise has been added also to the x and y-position of the centroid
