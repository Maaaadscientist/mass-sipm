import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import numpy as np

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


for channel in range(1,16):
  for position in range(16):
    for var in ["sigAmp","sigQ"]:
      for voltage in range(2,7):
        filtered_df = df.loc[(df['run_number'] == run_number) & (df['var'] == var)& (df['channel'] == channel) &
                       (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
                       (df['voltage'] == voltage)]
        print(channel, position, voltage,'\n')
        for value in filtered_df['mean']:
          print(value)
        for value in filtered_df['mean_error']:
          print(value)
        
        mean_list = [value for value in filtered_df['mean']]
        mean_error_list = [value for value in filtered_df['mean_error']]
        
        # Calculate peak distances
        peak_distances = np.diff(mean_list)
        
        # Calculate uncertainties
        uncertainties = np.sqrt(np.diff(mean_error_list)**2 + np.diff(mean_error_list)**2)
        
        # Generate x-axis labels
        x_labels = [f"Peak{i}-{i+1}" for i in range(len(peak_distances))]
        
        # Plotting
        x_values = np.arange(len(peak_distances))
        plt.errorbar(x_values, peak_distances, yerr=uncertainties, fmt='o', capsize=5)
        plt.xticks(x_values, x_labels)  # Set x-axis labels
        plt.ylabel('Distance')
        plt.title(f'Peak Distances of {var} (run{run_number} ch{channel} sipm{position} ov{voltage})')
        plt.savefig(f'{output_dir}/{var}_peak_distance_run{run_number}_ch{channel}_sipm{position}_ov{voltage}V.pdf')
        plt.clf()
        # Clear the lists before each iteration
        mean_list.clear()
        mean_error_list.clear()
      
