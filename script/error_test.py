import numpy as np
from scipy import stats

# Your data
x = np.array([21, 22, 23, 24, 25])
y = np.array([10, 12, 15, 11, 13])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Number of Monte Carlo simulations
num_simulations = 1000

# Create an array to store the x-intercept values from each simulation
x_intercepts = np.zeros(num_simulations)

# Perform Monte Carlo simulations
for i in range(num_simulations):
    # Perturb the y-coordinates based on the standard error
    y_perturbed = np.random.normal(y, std_err)
    
    # Perform linear regression with perturbed y-coordinates
    slope_perturbed, intercept_perturbed, _, _, _ = stats.linregress(x, y_perturbed)
    
    # Calculate x-intercept for the perturbed regression line
    x_intercepts[i] = -intercept_perturbed / slope_perturbed

# Calculate the mean and standard deviation of x-intercept values
x_intercept_mean = np.mean(x_intercepts)
x_intercept_std = np.std(x_intercepts)

# Calculate the ±1 sigma confidence interval for the x-intercept
x_intercept_err = x_intercept_std

# Print the results
print(f"x_intercept: {x_intercept_mean:.2f}")
print(f"x_intercept_err: {x_intercept_err:.2f}")

