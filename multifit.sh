#!/bin/bash

# Define the range for each ov value
python="/opt/homebrew/bin/python3.10"

# Loop over the desired parameters
#ov=1
sipmgr=1
root_type="tile"
file_path="main_run_0077"
run_info="main_run_0077"
output_file="newfit"

for ov in {1..2}; do
  for ch in {0..15}; do
    # Construct the input filename
    input_file="${file_path}/${run_info}_ov_${ov}.00_sipmgr_$(printf "%02d" $sipmgr)_${root_type}.root"
  
  
    # Calculate the bins and range based on ov value
    charge_range_start=-20
    charge_range_end=$((ov * 50 + 50))
    charge_bins=$((2 * (charge_range_end - charge_range_start)))

    amp_range_start=0
    amp_range_end=$((ov * 4 + 2))
    amp_bins=$((20 * (amp_range_end - amp_range_start)))
  
    # Construct and execute the command
    charge_fit_command="$python fit_peaks.py ${input_file} run77_ov${ov}_main_${root_type}_ch${sipmgr} sigQ_ch${ch} ${charge_bins} ${charge_range_start} ${charge_range_end} ${output_file}"
    echo "Executing command: ${charge_fit_command}"
    #$charge_fit_command
    amp_fit_command="$python fit_peaks.py ${input_file} run77_ov${ov}_main_${root_type}_ch${sipmgr} sigAmp_ch${ch} ${amp_bins} ${amp_range_start} ${amp_range_end} ${output_file}"
    echo "Executing command: ${amp_fit_command}"
    $amp_fit_command
  done
done

