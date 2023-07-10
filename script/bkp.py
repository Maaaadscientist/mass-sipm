import pandas as pd
import math
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from copy import deepcopy

def draw_linear_fit(x, x_with_bkv, y, y_err, intercept, slope, unit, serial_number, output_path):
    # convert input lists to numpy arrays
    x = np.array(x)
    xplus = np.array(x_with_bkv)
    y = np.array(y)
    y_err = np.array(y_err)

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
    ax.set_ylabel(f'{unit}')
    ax.legend()

    # Set the y-axis limits
    plt.ylim([0, np.max(y) * 1.1])

    # Save the plot as a PDF file
    filename = f'{output_path}/{unit}_{serial_number}.pdf'
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
    x_intercept = -intercept/slope
    
    return slope, intercept, x_intercept


if len(sys.argv) < 3:
    print("Usage: python prepare_jobs <input_file> <output_dir>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
input_file =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
eos_mgm_url = "root://junoeos01.ihep.ac.cn"
directory = "/tmp/tao-sipmtest"
output_dir = os.path.abspath(output_tmp)
#output_dir = "/junofs/users/wanghanwen/main_run_0075"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Find a certain element based on other column values
run_number = 109
#channel = 1
type_ = 'tile'
#position = 0
run_type = 'main'
#voltage = 1
#peak = 0
var_dict = {"sigAmp":"Amplitude", "sigQ":"Charge"}
unit_dict = {"sigAmp":"mV", "sigQ":"Q"}

# Filter the DataFrame based on the conditions
#filtered_df = df.loc[(df['run_number'] == run_number) & (df['channel'] == channel) &
#                     (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
#                     (df['voltage'] == voltage) & (df['peak'] == peak)]


# Get the dictionary of column names and their indexes
column_dict = {column: index for index, column in enumerate(df.columns)}

print(column_dict)
# Access the desired element

var = "sigQ"

peak_num = [2, 2, 2, 3, 3, 4, 4]
vols = [i for i in range(2,7)]
for position in range(16):
  for channel in range(1,16):
    for voltage in range(2,7):
      filtered_df = df.loc[(df['run_number'] == run_number) & (df['var'] == var)& (df['channel'] == channel) &
                     (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
                     (df['voltage'] == voltage)]
      #if len(filtered_df["peak"]) < peak_num[voltage]:
      if len(filtered_df["peak"]) < 4:
        print("position:", position, "channel:", channel, "ov:", voltage, "number of peaks", len(filtered_df["peak"]) )
        
