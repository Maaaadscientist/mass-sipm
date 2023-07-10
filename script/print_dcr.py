import pandas as pd
import os, sys
import re
import statistics
import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_weighted_mean(x, errors):
    weighted_sum = 0.0
    weight_sum = 0.0

    for i in range(len(x)):
        weight = 1.0 / (errors[i] ** 2 + 0.000000001)
        weighted_sum += x[i] * weight
        weight_sum += weight

    weighted_mean = weighted_sum / weight_sum
    return weighted_mean


def calculate_standard_deviation_square(lst):
    if len(lst) < 2:
        return 0  # handle list with less than 2 elements

    std_dev = statistics.stdev(lst)
    return std_dev * std_dev

def calculate_mean_error_square(errors):
    weight_sum = 0.0

    for i in range(len(errors)):
        weight_sum += 1.0 / (errors[i] ** 2 + 0.000000001)

    mean_error = math.sqrt(1.0 / weight_sum)
    return mean_error * mean_error
def square_root_sum(numbers):
    total = 0
    for num in numbers:
        total += math.sqrt(num)
    return total
csv_dir = sys.argv[1:]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
#output_dir = "/junofs/users/wanghanwen/main_run_0075"


# Find a certain element based on other column values
pattern = r"(\w+)_(\w+)_dcr_(\d+)"
name_match = re.match(pattern, csv_dir[0])
if name_match:
    print("match!!!!")
    run_number = int(name_match.group(3))    # "64"
#channel = 4
type_ = 'tile'
#position = 0
run_type = 'main'
#voltage = 1
#peak = 0
var = "dcrQ"
time_cut = 1
# Filter the DataFrame based on the conditions
#filtered_df = df.loc[(df['run_number'] == run_number) & (df['channel'] == channel) &
#                     (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
#                     (df['voltage'] == voltage) & (df['peak'] == peak)]

all_data = []
for filename in csv_dir:
        data = pd.read_csv(filename)
        all_data.append(data)

df = pd.concat(all_data, ignore_index=True)

# Get the dictionary of column names and their indexes
column_dict = {column: index for index, column in enumerate(df.columns)}

print(column_dict)
# Access the desired element


voltage_list = range(1,7)
for position in range(16):
  dcr_list = []
  dcr_err_list = []
  for voltage in voltage_list:
    tmp_list = []
    tmp_err_list = []
    for channel in range(1,17):
      if run_number == 109 and channel == 7:
        continue
      if run_number == 109 and position == 13 and channel ==4:
        continue
      if run_number == 109 and position == 13 and channel ==12:
        continue
      filtered_df = df.loc[(df['run_number'] == run_number) & (df['var'] == var)& (df['channel'] == channel) &
                     (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
                     (df['voltage'] == voltage) & (df['timecut'] == time_cut)]
      if len(filtered_df['events']) == 0:
        continue
      events_list = [events for events in filtered_df['events']]
      signal_list = [events for events in filtered_df['signal_events']]
      init_list = [events for events in filtered_df['init_events']]
      Nev = events_list[0]
      signal_unc_list = [events / 144. / (1054 * 8e-9 * Nev) for events in filtered_df['signal_events_unc']]
      dcr_N = signal_list[0] + sum(init_list[1:-1])
      dcr_rate = dcr_N / 144. / (1054 * 8e-9 * Nev)
      if channel == 2 and position == 0 and voltage == 2:
        print("results!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        print(dcr_rate)
      dcr_err = math.sqrt(square_root_sum(signal_unc_list))
      tmp_list.append(dcr_rate)
      tmp_err_list.append(dcr_err)
    dev_err = calculate_standard_deviation_square(tmp_list)
    dcr_list.append(calculate_weighted_mean(tmp_list,tmp_err_list))
    dcr_err_list.append(math.sqrt(calculate_mean_error_square(tmp_err_list)+dev_err))
  print(dcr_list)
  print(dcr_err_list)
  # Plotting the arrays 
  channels = np.array(voltage_list)

  # Plot with error bars
  plt.errorbar(channels, dcr_list, yerr=dcr_err_list, marker='o', label=f'tile {position}',capsize=4, linestyle = '--')


  # Adding labels and title 
  plt.xlabel('over voltage(V)')
  plt.ylabel('DCR (Hz/$mm^2$)')
  plt.title(f'Dark counting rate (run{run_number})')

  plt.ylim(0,200)
  # Adding legend 
  #plt.legend()
  # Adding a legend
  legend = plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

  # Adjusting the plot to accommodate the legend
  plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards


  # Customize the X-axis tick labels 
  plt.xticks(channels)
plt.savefig(f"dcr_tiles_run{run_number}.pdf")
plt.clf()


for voltage in voltage_list:
  dcr_list = []
  dcr_err_list = []
  for position in range(16):
    tmp_list = []
    tmp_err_list = []
    for channel in range(1,17):
      filtered_df = df.loc[(df['run_number'] == run_number) & (df['var'] == var)& (df['channel'] == channel) &
                     (df['type'] == type_) & (df['position'] == position) & (df['run_type'] == run_type) & 
                     (df['voltage'] == voltage) & (df['timecut'] == time_cut)]
      if len(filtered_df['events']) == 0:
        continue
      events_list = [events for events in filtered_df['events']]
      signal_list = [events for events in filtered_df['signal_events']]
      init_list = [events for events in filtered_df['init_events']]
      Nev = events_list[0]
      signal_unc_list = [events / 144. / (1054 * 8e-9 * Nev) for events in filtered_df['signal_events_unc']]
      dcr_N = signal_list[0] + sum(init_list[1:-1])
      dcr_rate = dcr_N / 144. / (1054 * 8e-9 * Nev)
      dcr_err = math.sqrt(square_root_sum(signal_unc_list))
      tmp_list.append(dcr_rate)
      tmp_err_list.append(dcr_err)
    dev_err = calculate_standard_deviation_square(tmp_list)
    dcr_list.append(calculate_weighted_mean(tmp_list,tmp_err_list))
    dcr_err_list.append(math.sqrt(calculate_mean_error_square(tmp_err_list)+dev_err))
  print(dcr_list)
  print(dcr_err_list)
  # Plotting the arrays 
  channels = np.arange(len(dcr_list))

  # Plot with error bars
  plt.errorbar(channels, dcr_list, yerr=dcr_err_list, marker='o', label=f'voltage {voltage}V',capsize=4, linestyle = '--')


  # Adding labels and title 
  plt.xlabel('tile number')
  plt.ylabel('DCR (Hz/$mm^2$)')
  plt.title(f'Dark counting rate (run{run_number})')

  plt.ylim(0,200)
  # Adding legend 
  #plt.legend()
  # Adding a legend
  legend = plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

  # Adjusting the plot to accommodate the legend
  plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards


  # Customize the X-axis tick labels 
  plt.xticks(channels)
plt.savefig(f"dcr_voltages_run{run_number}.pdf")
