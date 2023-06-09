import pandas as pd
import math
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from copy import deepcopy

def draw_linear_fit(fig, ax,ax2, x, x_with_bkv, y, y_err, intercept, slope, unit):
    # convert input lists to numpy arrays
    x = np.array(x)
    xplus = np.array(x_with_bkv)
    y = np.array(y)
    y_err = np.array(y_err)

    # Define the fitted line
    fit = slope * xplus + intercept
    # Plot the data points with error bars
    if unit == "Q":
      ax.errorbar(x, y, yerr=y_err, fmt='o',markersize=2, color='black', label='Data')

      # Plot the fitted line
      ax.plot(xplus, fit, color='red', label='Charge fit')
      # Set the limits for the twin y-axis
      ax.set_ylim([0, np.max(y) * 1.1])

      # Add labels and legend
      ax.set_xlabel('Voltage(V)')
      ax.set_ylabel(f'{unit}')
      ax.legend()
    if unit == "mV":
      ax2.errorbar(x, y, yerr=y_err, fmt='s', markersize=2,color='black', label='Data')

      # Plot the fitted line
      ax2.plot(xplus, fit, color='blue', label='Amplitude fit')

      # Set the limits for the twin y-axis
      ax2.set_ylim([0, np.max(y) * 1.1])
      # Add labels and legend
      ax2.set_xlabel('Voltage(V)')
      ax2.set_ylabel(f'{unit}')
      # Add legend for the additional arrays
      ax2.legend(loc='lower right')



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
run_number = 110
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


#vols = [i for i in range(2,7)]
for position in range(16):
  vbd_dict = {"sigAmp":[], "sigQ":[]}
  for channel in range(1,16):
    
    # Set up the plot
    fig, ax = plt.subplots()
    # Create a twin y-axis on the right side
    ax2 = ax.twinx()

    for var in ["sigAmp","sigQ"]:
      mean_diff_list = []
      mean_unc_list = []
      vols = []
      print(position,channel,'\n')
      for voltage in range(2,7):
        filtered_df = df.loc[(df['run_number'] == run_number) & (df['var'] == var)& (df['channel'] == channel) &
                       (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
                       (df['voltage'] == voltage)]
        
        mean_list = [value for value in filtered_df['mean']]
        mean_error_list = [value for value in filtered_df['mean_error']]
        
        # Calculate peak distances
        if var == "sigAmp":
          if len(mean_list) < 3:
            continue
          mean_diff_list.append(mean_list[2] - mean_list[1])
          mean_unc_list.append(math.sqrt(mean_error_list[2]*mean_error_list[2] + mean_error_list[1] * mean_error_list[1]))
        if var == "sigQ":
          if len(mean_list) < 2:
            continue
          mean_diff_list.append(mean_list[1] - mean_list[0])
          mean_unc_list.append(math.sqrt(mean_error_list[1]*mean_error_list[1] + mean_error_list[0] * mean_error_list[0]))
        # Clear the lists before each iteration
        #print("length of data:",len(mean_diff_list), "length of error:", len(mean_error_list))
        vols.append(voltage)
      slope, intercept, x_intercept =  linear_fit(vols,mean_diff_list,mean_unc_list)
      vols_plus = deepcopy(vols)
      vols_plus.insert(0, x_intercept)
      draw_linear_fit(fig, ax, ax2, vols,vols_plus ,mean_diff_list, mean_unc_list, intercept, slope, unit_dict[var]) 
      vbd_dict[var].append(x_intercept)
    # Set the y-axis limits

    # Save the plot as a PDF file
    filename = f'run{run_number}_ch{channel}_sipm{position}.pdf'
    plt.savefig(output_dir + "/" +filename)
    plt.clf()
  
  # Generate random values for the two arrays of breakdown voltage 
  size = 16  # Size of each array 
  breakdown_voltage_1 = vbd_dict["sigAmp"] 
  breakdown_voltage_2 = vbd_dict["sigQ"] 
   
  # Plotting the arrays 
  channels = np.arange(1, size + 1)  # X-axis values (channels) 
   
  plt.plot(channels, breakdown_voltage_1, marker='o', label='Vbd (Amp)') 
  plt.plot(channels, breakdown_voltage_2, marker='o', label='Vbd (Q)') 
   
  # Adding labels and title 
  plt.xlabel('Channel') 
  plt.ylabel('Breakdown Voltage') 
  plt.title(f'Breakdown Voltage Comparison (SiPM{position})') 
   
  # Adding legend 
  plt.legend() 
   
  # Customize the X-axis tick labels 
  plt.xticks(channels) 
   
  # Display the plot 
  plt.savefig(f'{output_dir}/breakdownVoltage_run{run_number}_sipm{position}.pdf') 
  plt.clf()
    
# Close the figure to save memory
plt.close()
      
