import numpy as np
import matplotlib.pyplot as plt
# Generate event to insert into noisy data
t = 100  # Duration of event in time points
# Derivative Gaussian event
event = np.diff(np.exp(-np.linspace(-2, 2, t+1)**2))
event = event/np.max(event)  # normalize to max=1
# Insert n of those events inside a data set
n = 30
onset = np.random.permutation(10000 - t)
onset = onset[0:n]

# Put events into data

signal = np.zeros(10000)
for event in range(n):
    signal[onset[event]:onset[event]+t] = event

# Add noise to signal (same magnitude as signal)
signal = signal + 0.5*np.random.randn(len(signal))
# plot data
plt.subplot(211)
plt.plot(signal)

# plot one event
plt.subplot(212)
plt.plot(range(0, t), signal[onset[3]:onset[3]+t])
plt.plot(range(0, t), event)
plt.show()
