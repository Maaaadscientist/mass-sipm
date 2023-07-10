import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Read the CSV file
data = pd.read_csv('test.csv')

# Group the data by run and tile
groups = data.groupby(['run', 'tile'])

# Initialize lists to store the breakdown voltages and coefficients
breakdown_voltages = []
coefficients = []

# Process each group separately
for group_name, group_data in groups:
    run, tile = group_name

    # Extract the voltage and gain data for the current group
    voltage = group_data['voltage'].values
    gain = group_data['gain'].values

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(voltage, gain)

    # Calculate the x-intercept (breakdown voltage)
    breakdown_voltage = -intercept / slope

    # Store the breakdown voltage and coefficient
    breakdown_voltages.append(breakdown_voltage)
    coefficients.append((slope, intercept))

    # Plot the linear fit
    plt.scatter(voltage, gain, label=f'Run {run}, Tile {tile}')
    plt.plot(voltage, slope * voltage + intercept)

    # Add breakdown voltage as text on the plot
    plt.text(0.95, 0.05, f'Breakdown Voltage: {breakdown_voltage:.2f}', 
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Set plot labels and title
plt.xlabel('Voltage')
plt.ylabel('Gain')
plt.title('Linear Fit of Gain at Different Voltages')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Print the breakdown voltages and coefficients
for i, (slope, intercept) in enumerate(coefficients):
    run, tile = groups.indices.keys()[i]
    print(f"Run {run}, Tile {tile}: Breakdown Voltage = {breakdown_voltages[i]:.2f}, Slope = {slope:.2f}, Intercept = {intercept:.2f}")

