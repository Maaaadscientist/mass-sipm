import numpy as np
import matplotlib.pyplot as plt

# Generate x values from 0 to 1
x = np.linspace(0, 1, 100)

# Calculate y values for exp(-x)
y_exp = np.exp(-x)

# Calculate y values for 1-x
y_linear = 1 - x

# Calculate residuals
residuals = y_exp - y_linear

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Plot the functions in the top subplot
ax1.plot(x, y_exp, label='exp(-x)')
ax1.plot(x, y_linear, label='1-x')
ax1.set_ylabel('y')
ax1.set_ylim([0, 1])
ax1.legend()

# Plot the residuals in the bottom subplot
ax2.plot(x, residuals, label='Residuals')
ax2.set_xlabel('x')
ax2.set_ylabel('Residuals')
ax2.axhline(0, color='black', linewidth=0.5)  # Add a horizontal line at y=0
ax2.set_ylim([0,0.1])
ax2.grid(True)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.3)

# Display the plot
plt.show()

