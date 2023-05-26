import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def draw_linear_fit(x, x_with_bkv, y, y_err, intercept, x_intercept, slope, serial_number):
    # convert input lists to numpy arrays
    x = np.array(x)
    xplus = np.array(x_with_bkv)
    y = np.array(y)
    y_err = np.array(y_err)
    breakdown_voltage = x_intercept

    # Define the fitted line
    fit = slope * xplus + intercept

    # Set up the plot
    fig, ax = plt.subplots()

    # Plot the data points with error bars
    ax.errorbar(x, y, yerr=y_err, fmt='o', color='black', label='Data')

    # Plot the fitted line
    ax.plot(xplus, fit, color='red', label='Linear fit')

    # Add labels and legend
    ax.set_xlabel('Voltage(V)')
    ax.set_ylabel('Gain')
    ax.legend()

    # Set the y-axis limits
    plt.ylim([0, np.max(y) * 1.1])

     # Add breakdown voltage as text on the plot
    plt.text(0.95, 0.05, f'Breakdown Voltage: {breakdown_voltage:.2f}', 
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Set the plot title
    plt.title(f'Linear Fit of Gain at Different Voltages\nRun: {serial_number[0]}, Tile: {serial_number[1]}')

    # Save the plot as a PDF file
    filename = f'gain_run{serial_number[0]}_tile{serial_number[1]}.pdf'
    plt.savefig(filename)

    # Close the figure to save memory
    plt.close()

def linear_fit(x_list, y_list, y_err_list):
    # convert input lists to numpy arrays
    x = np.array(x_list)
    y = np.array(y_list)

    # perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # calculate x-intercept
    x_intercept = -intercept / slope

    return slope, intercept, x_intercept

# Read the CSV file
data = pd.read_csv('test.csv')

# Group the data by tile
groups = data.groupby(['run','tile'])

# Initialize lists to store the breakdown voltages
breakdown_voltages = []

# Process each group separately
for group_name, group_data in groups:
    run, tile = group_name
    
    # Extract the voltage and gain data for the current group
    voltage = group_data['voltage'].values
    gain = group_data['gain'].values

    # Perform linear fit
    slope, intercept, x_intercept = linear_fit(voltage, gain, np.zeros_like(gain))

    # Store the breakdown voltage
    breakdown_voltages.append(x_intercept)

    # Draw linear fit plot
    x_with_bkv = np.concatenate((voltage, [x_intercept]))
    draw_linear_fit(voltage, x_with_bkv, gain, np.zeros_like(gain), intercept, x_intercept, slope, (run, tile))


for i,vbd in enumerate(breakdown_voltages):
    print(vbd)
