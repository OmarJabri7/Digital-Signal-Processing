#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:22:53 2021

@author: omar

Median Filter to remove Spiked (Peak) noise
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#Create noisy rando signal (Brownian Noise)

plt.close('all')

n = 2000
signal = np.cumsum(np.random.randn(n))
plt.plot(signal)
plt.xlabel("Time (s)")
plt.ylabel("Gain")

#To create peaks, select random noisy samples and replace them with large nbrs!
noise_proportion = 0.05

noise_samples = np.random.permutation(n)
noise_samples = noise_samples[0:int(n*noise_proportion)]

#Replace noise points at proportion with peaks
signal[noise_samples] = 50 + np.random.rand(len(noise_samples))*100

plt.plot(signal)
plt.xlabel("Time (s)")
plt.ylabel("Gain")
plt.legend(["White Noise", "Peak Noise"])
plt.figure()

plt.hist(signal,100)
plt.figure()

visual_threshold = 48 #Changes cause random samples, this can be searched using a certain heuristic, to find the best threshold

#Samples above threshold (where less noise)
samples_above = np.where(signal>visual_threshold)[0]
#Can filter over all signal
#samples_above = range(0,n) #Uncomment to filter over all signal
#This is not advisable since we would lose enough information about the signal, we want to preserve information while filtering.

#Loop through these thresholded samples and set their values to the median of k
#Where k is the window:
k = 20

#Set filtered signal
filt_sig = deepcopy(signal)

for t in range(len(samples_above)):
    lower_bound = np.max((0,samples_above[t]-k)) #If this is the first data point, -k cannot be indexed, so take 0
    upper_bound = np.min((samples_above[t] + k,n)) #Same as above but vice versa.
    
    filt_sig[samples_above[t]] = np.median(signal[lower_bound:upper_bound])
    
#What this does is, replace the values above the threshold (samples with large values i.e. outliers) with the median of a
#neighborhood of k (here 20).
    
plt.plot(range(0,n),signal)
plt.plot(range(0,n),filt_sig)
plt.xlabel("Time (s)")
plt.ylabel("Gain")
plt.legend(["Peak Noise","Median Filtered"])
plt.show()   


