# Fitting a line through a time series, remove global trends from a signal.
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

# Create signal with trend (linear)
n = 2000
signal = np.cumsum(np.random.randn(n)) + np.linspace(-30, 30, n)
# Remove linear trend
detrending_signal = sp.detrend(signal)
# Plot both signals together
plt.plot(signal)
plt.plot(detrending_signal)
plt.xlabel("Time (s)")
plt.ylabel("Gain")
plt.legend([f"Original Signal with mean: {np.mean(signal)}",
            f"Denoised signal with mean: {np.mean(detrending_signal)}"])
plt.show()
# Notice how the algorithm removes the trend of the signal (either going up or
# down or diagonally)
