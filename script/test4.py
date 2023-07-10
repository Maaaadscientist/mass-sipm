import matplotlib.pyplot as plt
import numpy as np

# Generate random values for the two arrays of breakdown voltage
np.random.seed(0)  # Set a seed for reproducibility
size = 15  # Size of each array
breakdown_voltage_1 = np.random.uniform(0, 10, size)
breakdown_voltage_2 = np.random.uniform(0, 10, size)

# Plotting the arrays
channels = np.arange(1, size + 1)  # X-axis values (channels)

plt.plot(channels, breakdown_voltage_1, marker='o', label='Array 1')
plt.plot(channels, breakdown_voltage_2, marker='o', label='Array 2')

# Adding labels and title
plt.xlabel('Channel')
plt.ylabel('Breakdown Voltage')
plt.title('Breakdown Voltage Comparison')

# Adding legend
plt.legend()

# Customize the X-axis tick labels
plt.xticks(channels)

# Display the plot
plt.show()

