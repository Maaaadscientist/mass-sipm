#!/usr/bin/env python3
import os, sys
from copy import deepcopy

if len(sys.argv) < 3:
    print("Usage: python prepare_jobs <input_dir> <output_dir>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
input_dir = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
run_info = input_dir.split("/")[-1]
run_number = int(run_info.split("_")[-1])

if not os.path.isdir(output_dir + "/jobs"):
    os.makedirs(output_dir + "/jobs")
if not os.path.isdir(output_dir + "/plots"):
    os.makedirs(output_dir + "/plots")
if not os.path.isdir(output_dir + "/hists"):
    os.makedirs(output_dir + "/hists")
if not os.path.isdir(output_dir + "/dcr"):
    os.makedirs(output_dir + "/dcr")
template_hist = """for ov in {1..6}; do
  for ch in {0..16}; do
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
    charge_fit_command="$python draw_histos.py ${input_file} run${run_number}_ov${ov}_main_${root_type}_ch${sipmgr} sigQ_ch${ch} ${charge_bins} ${charge_range_start} ${charge_range_end} ${output_file}/hists"
    echo "Executing command: ${charge_fit_command}"
    $charge_fit_command
    amp_fit_command="$python draw_histos.py ${input_file} run${run_number}_ov${ov}_main_${root_type}_ch${sipmgr} sigAmp_ch${ch} ${amp_bins} ${amp_range_start} ${amp_range_end} ${output_file}/hists"
    echo "Executing command: ${amp_fit_command}"
    $amp_fit_command
  done
done"""

template_fit = """for ov in {1..6}; do
  for ch in {0..16}; do
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
    charge_fit_command="$python fit_peaks.py ${input_file} run${run_number}_ov${ov}_main_${root_type}_ch${sipmgr} sigQ_ch${ch} ${charge_bins} ${charge_range_start} ${charge_range_end} ${output_file}/plots"
    echo "Executing command: ${charge_fit_command}"
    $charge_fit_command
    amp_fit_command="$python fit_peaks.py ${input_file} run${run_number}_ov${ov}_main_${root_type}_ch${sipmgr} sigAmp_ch${ch} ${amp_bins} ${amp_range_start} ${amp_range_end} ${output_file}/plots"
    echo "Executing command: ${amp_fit_command}"
    $amp_fit_command
  done
done"""
template_dcr = """for ov in {1..6}; do
  for ch in {0..16}; do
    # Construct the input filename
    input_file="${file_path}/${run_info}_ov_${ov}.00_sipmgr_$(printf "%02d" $sipmgr)_${root_type}.root"


    # Calculate the bins and range based on ov value
    charge_range_start=-10
    charge_range_end=$((ov * 40 + 10))
    charge_bins=$((1 * (charge_range_end - charge_range_start)))

    # Construct and execute the command
    charge_fit_command="$python get_dcr.py ${input_file} run${run_number}_ov${ov}_main_${root_type}_ch${sipmgr} dcrQ_ch${ch} ${charge_bins} ${charge_range_start} ${charge_range_end} ${output_file}/dcr"
    echo "Executing command: ${charge_fit_command}"
    $charge_fit_command
  done
done"""

for ch in range(1,17):
  script  = ''
  script += '#!/bin/bash\n'
  script += f'directory="{output_dir}"\n'
  script += 'if [ ! -d "$directory" ]; then\n'
  script += '  echo "Output directory does not exist. Creating directory..."\n'
  script += '  mkdir -p "$directory"\n'
  script += '  echo "Directory created."\n'
  script += 'else\n'
  script += '  echo "Directory already exists."\n'
  script += 'fi\n'
  script += 'cd $directory\n'
  script += 'sleep 3\n'
  script += 'cp /junofs/users/wanghanwen/sipm-massive/env_lcg.sh .\n'
  script += 'cp /junofs/users/wanghanwen/sipm-massive/fit_peaks.py .\n'
  script += 'cp /junofs/users/wanghanwen/sipm-massive/draw_histos.py .\n'
  script += 'cp /junofs/users/wanghanwen/sipm-massive/get_dcr.py .\n'
  script += '. ./env_lcg.sh\n'
  script += 'expected_string="/cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/bin/python3"\n'
  script += 'python=$(which python3)\n'
  script += """
if [ "$python" = "$expected_string" ]; then
    echo "Command returned the expected string: python=$expected_string"
else
    echo "Command did not return the expected string."
    cp /junofs/users/wanghanwen/sipm-massive/env_lcg.sh .
    . ./env_lcg.sh
fi\n"""
  script += 'sleep 10\n'
  script += f'sipmgr={ch}\n'
  script += 'root_type="tile"\n'
  script += f'file_path="{input_dir}"\n'
  script += f'run_info="{run_info}"\n'
  script += f'run_number="{run_number}"\n'
  script += f'output_file="{output_dir}"\n'
  for job_type in ["fit","hist","dcr"]:
    with open(f'{output_dir}/jobs/{job_type}_job_{ch}.sh','w') as file_tmp:
      script_tmp = deepcopy(script)
      if job_type == "hist":
        script_tmp += template_hist
      elif job_type =="fit":
        script_tmp += template_fit
      elif job_type =="dcr":
        script_tmp += template_dcr
 
      script_tmp += '\n'
      script_tmp += 'cd -\n'
      file_tmp.write(script_tmp)

os.system(f"chmod +x {output_dir}/jobs/*.sh")
os.system(f"cp submit_fit_jobs.sh {output_dir}")
os.system(f"chmod +x {output_dir}/submit_fit_jobs.sh ")
os.system(f"cp combine_csv.py {output_dir}")
