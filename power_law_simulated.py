import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a*np.power(x, b)

# Generate dummy dataset
x_dummy = np.linspace(start=1, stop=1000, num=100)
y_dummy = power_law(x_dummy, 1, 0.5)

# Add noise from a Gaussian distribution
noise = 1.5*np.random.normal(size=y_dummy.size)
y_dummy = y_dummy + noise

# Set the x and y-axis scaling to logarithmic
fig, ax = plt.subplots()

# Fit the dummy power-law data
pars, cov = curve_fit(f=power_law, xdata=x_dummy, ydata=y_dummy, p0=[0, 0], bounds=(-np.inf, np.inf))
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))
# Calculate the residuals
res = y_dummy - power_law(x_dummy, *pars)
print(res)

# Plot power law
plt.plot(y_dummy)
plt.plot(power_law(x_dummy, *pars))
plt.show()

# ax.set_xscale('log')
# ax.set_yscale('log')
# # Edit the major and minor tick locations of x and y axes
# ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
# ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
# # Set the axis limits
# ax.set_xlim(10, 1000)
# ax.set_ylim(1, 100)

