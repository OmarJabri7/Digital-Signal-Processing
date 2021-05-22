#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:52:55 2021

@author: omar

Denoising EMG signal using TKEO
"""
import scipy.io
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


#Extract EMG Mat File

emg_data = scipy.io.loadmat('emg_data.mat')

#Save contents in vars

emg = emg_data["emg"][0]

emg_time = emg_data["emgtime"][0]

fs = emg_data["fs"]

#Sample points
n = len(emg)

#Initialize denoised signal
filt_emg = deepcopy(emg)

#Equation for TKEO denoiser: y(t) = x(t)**2 - x(t - 1)*x(t + 1)

for s in range(1,n - 1):
    filt_emg[s] = emg[s]**2 - (emg[s - 1]*emg[s + 1])
    
#N.B.: It is not good to use loops, so use vectorization!
filt_emg = deepcopy(emg)
filt_emg[1:n-1] = emg[1:n-1]**2 - (emg[0:n-2]*emg[2:n])

#Plot filtered vs noisy EMG
plt.plot(emg_time,emg/np.max(emg))
plt.plot(emg_time,filt_emg/np.max(filt_emg))
plt.xlabel("Time (s")
plt.ylabel("Gain")
plt.legend("Original","Filtered")
plt.figure()

#To be able to correctly compare the two signals, we need to calculate their z score
#and plot according to that score (both signals have different scales)

#Calculate time at which to calculate z score from, not over all signal

time_z = np.argmin(emg_time**2)

zscore_orig = (emg - np.mean(emg[0:time_z]))/np.std(emg[0:time_z])

zscore_filt = (filt_emg - np.mean(filt_emg[0:time_z]))/np.std(filt_emg[0:time_z])

#Plot filtered vs noisy EMG z score
plt.plot(emg_time,zscore_orig)
plt.plot(emg_time,zscore_filt)
plt.xlabel("Time (s")
plt.ylabel("Gain")
plt.legend("Original","Filtered")
plt.show()

    

