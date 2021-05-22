#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:18:28 2021

@author: omar

Running Mean Filter
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import *
import copy

def mean_filter(signal):
    #Initialize filtered signal
    filt_sig = np.zeros(len(signal))
    for s in range(k+1, n-k-1):
        filt_sig[s] = np.mean(signal[s - k:s + k])
    return filt_sig

#Create signal
sampling_rate = 1000 # Unit: Hz
time = np.arange(0,3,1/sampling_rate )# 3 seconds
n = len(time) #Sample size
poles = 15 #Poles for random interpolation

#Noise level, measured in std
noise_level = 5

#Interpolate neighboring discrete samples
amplitude = np.interp(np.linspace(0,poles,n),np.arange(0,poles),np.random.rand(poles)*30)
noise = noise_level*np.random.rand(n) #Adjust noise over all samples randomly
signal = amplitude + noise #Add noise to signal

#Plot without noise: @ampltiude
plt.plot(time,amplitude)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.figure()

#Plot with noise: @signal
plt.plot(time,signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.figure()

#Mean Filter Equation => filt_sig = ((2*k + 1)**(-1))* sum(signal)
#Set K, you can modify K as smoothing level parameter: High K, Higher smoothing & otherwise
k = 20
#Apply mean filter on @signal
filt_sig = mean_filter(signal)
#Calculate window size in milli seconds
window_size = 1000*(k*2 + 1)/sampling_rate #Window size in points is the same in ms

#Plot
plt.plot(time,signal)
plt.plot(time,filt_sig)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(["Original", "Mean Filtered"])