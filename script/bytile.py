import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Read the CSV file
data = pd.read_csv('test.csv')

# Group the data by tile
groups = data.groupby('tile')

# Initialize lists to store the breakdown voltages and coefficients
breakdown_voltages = []
coefficients = []

# Process each group separately
for group_name, group_data in groups:
    tile = group_name

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
    plt.scatter(voltage, gain, label=f'Tile {tile}')
    plt.plot(voltage, slope * voltage + intercept)

# Set plot labels and title
plt.xlabel('Voltage')
plt.ylabel('Gain')
plt.title('Linear Fit of Gain at Different Voltages by Tiles')

# Add a legend
plt.legend()

# Add breakdown voltages as text on the plot
text_offset = 0.05
for i, voltage in enumerate(breakdown_voltages):
    text_y = 0.9 - i * text_offset
    plt.text(1.05, text_y, f'Tile {i+1}: {voltage:.2f}', 
             verticalalignment='center', horizontalalignment='left',
             transform=plt.gca().transAxes, fontsize=10)

# Set up a separate pad on the right for breakdown voltage text
plt.annotate('', xy=(1.15, 1), xycoords='axes fraction', xytext=(1.15, 0), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='-', color='black'))
plt.text(1.2, 0.5, 'Breakdown Voltage', rotation=90, verticalalignment='center', horizontalalignment='left',
         transform=plt.gca().transAxes, fontsize=12)

# Extend the x-axis limits
x_min, x_max = plt.xlim()
plt.xlim(x_min -1.5, x_max + 1.5)

# Set the y-axis range to start from 0
plt.ylim(0, plt.ylim()[1])

# Show the plot
plt.show()

# Print the breakdown voltages and coefficients
for i, (slope, intercept) in enumerate(coefficients):
    tile = groups.indices.keys()[i]
    print(f"Tile {tile}: Breakdown Voltage = {breakdown_voltages[i]:.2f}, Slope = {slope:.2f}, Intercept = {intercept:.2f}")

