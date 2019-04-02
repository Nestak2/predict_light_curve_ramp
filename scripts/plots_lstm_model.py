
# coding: utf-8

# ## A script that loads the saved model, does a prediction of a sequence of data with it and plots the prediction

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.python.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# In[16]:


# the lenght of the lightcurve in data points
len_lc = 690
# the length of a sequence of datapoints used as input
len_seq = 100
# the shift along the lightcurve with respect to its start
shift_pred = 130 #50*1/690.

model = load_model('/Users/narsenov/Documents/synced_docs/UCL internship/code/simulations_directories/2019-03-28_12_09_gith_option1_lenseq=100_epochs=10_traininglcs=900_batch_s=64/savemodel.h5')
# model = load_model("/Users/narsenov/Documents/synced_docs/UCL internship/code/simulations_directories/2019-03-03_18_38_gith_option1_lenseq=100_epochs=10_traininglcs=1000/savemodel.h5") #, custom_objects={'loss_mse_warmup': loss_mse_warmup} 

# load as input data the original light curve, which the network hasn't seen yet (it was trained on augmented data)
data_full = np.loadtxt('/Users/narsenov/Documents/synced_docs/UCL internship/code/data_processing_directories/2019-03-20_11_52_many_parameters_from_IRAF_star1_bin=True/output_file.txt', skiprows=1, delimiter=",")[-690:, [1,2,12]]
# the line below adds as a column a proxy for the time (690 consequent points in time)
data_full = np.c_[np.arange(690),data_full]

# data_full = np.loadtxt("/Users/narsenov/Documents/synced_docs/UCL internship/code/lightcurves_augmented_lenved=4_fromfinalramp_Aug7.txt", skiprows=0, delimiter=",")[-690:]
# before predicting, we need to scale the data exactly the way it was scaled for training 
sc_data = MinMaxScaler(feature_range = (10**(-10), 1.0))
data_scaled = sc_data.fit_transform(data_full)
X_seq_singleseq = data_scaled[-len_lc+shift_pred:-len_lc+shift_pred+len_seq]

predict_singleseq = model.predict(np.array(X_seq_singleseq).reshape((1,len_seq,data_full.shape[1])))
predict_singleseq = np.reshape(predict_singleseq, (-1,data_full.shape[1]))
# we need to inverse scale the prediction to obtain the real values
predict_singleseq_inv = sc_data.inverse_transform(predict_singleseq)


plt.figure(figsize=(20,12))

plt.scatter(data_full[:,0], data_full[:,3], s=3, label="full light curve", color="lightblue")
plt.scatter(predict_singleseq_inv[:,0], predict_singleseq_inv[:,3], s=3, label="predicted sequence", color="red")

plt.legend(prop={'size': 14})

