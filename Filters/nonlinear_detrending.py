# What happens if you have non linear trends in the data?
# Use non linear trending. Such as fluctuations in the sensor.
# This trend would be known as a slow drift in the signal.
# You can fit the signal with a polynomial and fit it to the signal, resulting
# in a normal drift.

import numpy as np
import matplotlib.pyplot as plt
from scipy import *

# polynomial intuition

order = 8
x = np.linspace(-15, 15, 100)

y = np.zeros(len(x))

for i in range(order+1):
    y = y + np.random.randn(1)*x**i

plt.plot(x, y)
plt.title('Order-%d polynomial' % order)
plt.figure()

# Signal with polynomial artifacts

n = 10000
t = range(n)
k = 10

slowdrift = np.interp(np.linspace(1, k, n),
                      np.arange(0, k), 100*np.random.randn(k))
signal = slowdrift + 20*np.random.randn(n)

# plot
plt.plot(t, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.figure()
# fit a 3-order polynomial

# polynomial fit (returns coefficients)
# Change order, because some are better than others!
p = np.polyfit(t, signal, 3)

# predicted data is evaluation of polynomial
yHat = np.polyval(p, t)

# compute residual (the cleaned signal)
residual = signal - yHat


# now plot the fit (the function that will be removed)
plt.plot(t, signal, 'b', label='Original')
plt.plot(t, yHat, 'r', label='Polyfit')
plt.plot(t, residual, 'k', label='Filtered signal')

plt.legend()
plt.figure()
# To determine the optimal order, use Bayes information criteria!
# Evaluate the fit of a model to the dataset
# Number of orders
orders = range(5, 40)
sse = np.zeros(len(orders))

for s in range(len(orders)):
    # Fit polynomial model on time-series data and predict y
    y_hat = np.polyval(np.polyfit(t, signal, orders[s]), t)

    sse[s] = np.sum((y_hat - signal)**2)/n

# Calculate Bayes Criterion of Information


bic = n*np.log(sse) + orders*np.log(n)
best_order = np.min(bic)
idx = np.argmin(bic)

# plot the BIC
plt.plot(orders, bic, 'ks-')
plt.plot(orders[idx], best_order, 'ro')
plt.xlabel('Polynomial order')
plt.ylabel('Bayes information criterion')
plt.figure()

# now repeat filter for best (smallest) BIC

# polynomial fit
polycoefs = np.polyfit(t, signal, orders[idx])

# estimated data based on the coefficients
y_hat = np.polyval(polycoefs, t)

# filtered signal is residual
filt_sig = signal - y_hat


# plotting
plt.plot(t, signal, 'b', label='Original')
plt.plot(t, y_hat, 'r', label='Polynomial fit')
plt.plot(t, filt_sig, 'k', label='Filtered')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
