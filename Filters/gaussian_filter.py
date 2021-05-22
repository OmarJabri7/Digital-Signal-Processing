#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:18:28 2021

@author: omar

Gaussian Smoothing Filter
"""

# I. Gaussian Density Function: G = exp((-4*ln(2)*(t**2))/w**2) => t: time (good to be centered at 0)
# This formulation allows you to specify w, width of gaussian: Full-width at half maximum.

from copy import deepcopy
from scipy import *
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def gaussian_filter(gaussian, signal, n):
    filt_sig = deepcopy(signal)
    for s in range(k + 1, n - k - 1):
        filt_sig[s] = np.sum(signal[range(s - k, s + k)]*gaussian)
    return filt_sig


# Create signal
sampling_rate = 1000  # Unit: Hz
time = np.arange(0, 3, 1/sampling_rate)  # 3 seconds
n = len(time)  # Sample size
poles = 15  # Poles for random interpolation

# Noise level, measured in std
noise_level = 5

# Interpolate neighboring discrete samples
amplitude = np.interp(np.linspace(0, poles, n), np.arange(
    0, poles), np.random.rand(poles)*30)
noise = noise_level*np.random.rand(n)  # Adjust noise over all samples randomly
signal = amplitude + noise  # Add noise to signal

# Full width half maximum
w = 25

# Normalized time vector in indices
# Changing k will lead to a better approximation with amplitude.
k = 100
gtime = 1000*(np.arange(-k, k))/(sampling_rate)

gaussian = np.exp((-4*np.log(2)*(gtime**2))/w**2)

# Empirically find full width at half maximum fwhm

pstPeakHalf = k + np.argmin((gaussian[k:]-.5)**2)
prePeakHalf = np.argmin((gaussian-.5)**2)

empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]

# Plot the Gaussian
plt.plot(gtime, gaussian, 'ko-')
plt.plot([gtime[prePeakHalf], gtime[pstPeakHalf]], [
         gaussian[prePeakHalf], gaussian[pstPeakHalf]], 'm')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.figure()

# Normalize gaussian, by the sum of all points in gaussian => filtered signal is in the same units
# as original signal

gaussian = gaussian / np.sum(gaussian)

# Filter noisy signal
filt_sig = gaussian_filter(gaussian, signal, n)

# Plot
plt.plot(time, signal)
plt.plot(time, filt_sig)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(["Original", "Gaussian Filtered"])
plt.figure()

# Notice edge efefcts, because starting at k + 1 with convolution. Lowering
# K will result in less edge effects

# II. Smooth Spiked Time Series with gaussian
# Number of spikes in interval
spikes = 300

# Generate inter-spike intervals

isi = np.round(np.exp(np.random.randn(spikes, 1))*10)

# Generate spike time-series
time_series = np.zeros(int(sum(isi)))

for i in range(0, spikes):
    time_series[int(np.sum(isi[0:i]))] = 1

# Filter spiked signal
filt_spiked_sig = gaussian_filter(gaussian, isi, len(isi))

# Plot filtered and spiked
plt.plot(isi)
plt.plot(filt_spiked_sig)
plt.xlabel("Time exp")
plt.ylabel("Amplitude")
plt.legend(["Spiked", "Gaussian Filtered"])
plt.figure()
