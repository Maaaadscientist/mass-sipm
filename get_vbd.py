import pandas as pd
import math
import os, sys
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from copy import deepcopy

def get_diff_list(lst):
    if len(lst) < 2:
        return []
    tmp_lst = []
    for i in range(len(lst) - 1):
        tmp_lst.append(lst[i+1] - lst[i])
    return tmp_lst

def get_diff_error_list(lst):
    if len(lst) < 2:
        return []
    tmp_lst = []
    for i in range(0, len(lst) - 1):
        tmp_lst.append(math.sqrt(lst[i+1]*lst[i+1] + lst[i]*lst[i]))

    return tmp_lst

def calculate_weighted_mean(x, errors):
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for i in range(len(x)):
        weight = 1.0 / (errors[i] ** 2)
        weighted_sum += x[i] * weight
        weight_sum += weight
    
    weighted_mean = weighted_sum / weight_sum
    return weighted_mean

def calculate_mean_error_square(errors):
    weight_sum = 0.0
    
    for i in range(len(errors)):
        weight_sum += 1.0 / (errors[i] ** 2)
    
    mean_error = math.sqrt(1.0 / weight_sum)
    return mean_error * mean_error

def calculate_standard_deviation_square(lst):
    if len(lst) < 2:
        return 0  # handle list with less than 2 elements

    std_dev = statistics.stdev(lst)
    return std_dev * std_dev

def calculate_mean(lst):
    if len(lst) == 0:
        return 0  # handle empty list case

    mean_value = statistics.mean(lst)
    return mean_value


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

def linear_fit_bootstrap(x_list, y_list, y_err_list, n_bootstrap=1000):
    # convert input lists to numpy arrays
    x = np.array(x_list)
    y = np.array(y_list)
    y_err = np.array(y_err_list)

    # perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # calculate x-intercept
    x_intercept = -intercept / slope

    # bootstrap resampling
    bootstrap_slopes = []
    bootstrap_intercepts = []
    bootstrap_x_intercepts = []

    for _ in range(n_bootstrap):
        # create bootstrap sample by varying each element of y_list separately
        bootstrap_y = np.array([y[i] + np.random.choice([-1, 1]) * y_err[i] for i in range(len(y))])

        # perform linear regression on the bootstrap sample
        bootstrap_slope, bootstrap_intercept, _, _, _ = stats.linregress(x, bootstrap_y)

        # calculate x-intercept for bootstrap sample
        bootstrap_x_intercept = -bootstrap_intercept / bootstrap_slope

        # store bootstrap results
        bootstrap_slopes.append(bootstrap_slope)
        bootstrap_intercepts.append(bootstrap_intercept)
        bootstrap_x_intercepts.append(bootstrap_x_intercept)

    # calculate uncertainties using bootstrap results
    slope_std_err = np.std(bootstrap_slopes)
    intercept_std_err = np.std(bootstrap_intercepts)
    x_intercept_std_err = np.std(bootstrap_x_intercepts)

    return (
        slope,
        intercept,
        x_intercept,
        slope_std_err,
        intercept_std_err,
        x_intercept_std_err,
    )


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
  vbd_err_dict = {"sigAmp":[], "sigQ":[]}
  for channel in range(1,17):
    
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
        if position == 1 and voltage < 3:
            continue
        filtered_df = df.loc[(df['run_number'] == run_number) & (df['var'] == var)& (df['channel'] == channel) &
                       (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
                       (df['voltage'] == voltage)]
        
        mean_list = [value for value in filtered_df['mean']]
        mean_error_list = [value for value in filtered_df['mean_error']]
        
        # Calculate peak distances
        if var == "sigAmp":
          if len(mean_list) < 3:
            continue
          mean_list.pop(0)
          diff_list = get_diff_list(mean_list)
          diff_error_list = get_diff_error_list(mean_error_list)
          diff_value = calculate_weighted_mean(diff_list, diff_error_list) 
          diff_error = math.sqrt(calculate_mean_error_square(diff_error_list) + calculate_standard_deviation_square(diff_list))
          #diff_error = math.sqrt(calculate_standard_deviation_square(mean_list))
          mean_diff_list.append(diff_value)
          mean_unc_list.append(diff_error)
        if var == "sigQ":
          if len(mean_list) < 2:
            continue
          diff_list = get_diff_list(mean_list)
          diff_error_list = get_diff_error_list(mean_error_list)
          diff_value = calculate_weighted_mean(diff_list, diff_error_list) 
          diff_error = math.sqrt(calculate_mean_error_square(diff_error_list) + calculate_standard_deviation_square(diff_list))
          #diff_error = math.sqrt(calculate_standard_deviation_square(mean_list))
          mean_diff_list.append(diff_value)
          mean_unc_list.append(diff_error)
        # Clear the lists before each iteration
        #print("length of data:",len(mean_diff_list), "length of error:", len(mean_error_list))
        vols.append(voltage)
      #slope, intercept, x_intercept =  linear_fit(vols,mean_diff_list,mean_unc_list)
      slope, intercept, x_intercept , slope_err, intercept_err, x_intercept_err =  linear_fit_bootstrap(vols,mean_diff_list,mean_unc_list, 2000)
      vols_plus = deepcopy(vols)
      vols_plus.insert(0, x_intercept)
      draw_linear_fit(fig, ax, ax2, vols,vols_plus ,mean_diff_list, mean_unc_list, intercept, slope, unit_dict[var]) 
      vbd_dict[var].append(x_intercept)
      vbd_err_dict[var].append(x_intercept_err)
    # Set the y-axis limits

    # Save the plot as a PDF file
    filename = f'run{run_number}_ch{channel}_sipm{position}.pdf'
    plt.savefig(output_dir + "/" +filename)
    plt.clf()
  
  # Generate random values for the two arrays of breakdown voltage 
  size = 16  # Size of each array 
  breakdown_voltage_1 = vbd_dict["sigAmp"] 
  breakdown_voltage_2 = vbd_dict["sigQ"] 
  breakdown_voltage_err_1 = vbd_err_dict["sigAmp"] 
  breakdown_voltage_err_2 = vbd_err_dict["sigQ"] 
   
  # Plotting the arrays 
  channels = np.arange(1, size + 1)  # X-axis values (channels) 
   
  #plt.plot(channels, breakdown_voltage_1, marker='o', label='Vbd (Amp)') 
  #plt.plot(channels, breakdown_voltage_2, marker='o', label='Vbd (Q)') 
  # Plot with error bars
  plt.errorbar(channels, breakdown_voltage_1, yerr=breakdown_voltage_err_1, marker='o', label='Vbd (Amp)',capsize=5 , color = 'darkcyan')
  plt.errorbar(channels, breakdown_voltage_2, yerr=breakdown_voltage_err_2, marker='o', label='Vbd (Q)',capsize=5,color ='maroon')

  # Set y-axis range
  plt.ylim(-2, 0)
   
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
      
