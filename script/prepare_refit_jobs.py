#!/usr/bin/env python3
import os, sys

if len(sys.argv) < 3:
    print("Usage: python prepare_signal_jobs.py <input_dir> <output_dir> <binary_path>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
   binary_tmp = sys.argv[3]
input_dir = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
binary_path = os.path.abspath(binary_tmp)
parrent_path = "/".join(binary_path.split("/")[0:-2])
run_info = input_dir.split("/")[-1]
run_number = int(run_info.split("_")[-1])

if not os.path.isdir(output_dir + "/jobs"):
    os.makedirs(output_dir + "/jobs")
if not os.path.isdir(output_dir + "/logs/cpuInfo"):
    os.makedirs(output_dir + "/logs/cpuInfo")

for ch in range(1, 17):
    for ov in range(1, 7):
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
        script += f'cp {parrent_path}/script/charge_fit.py .\n'
        script += f'cp {parrent_path}/env_lcg.sh .\n'
        script += '. ./env_lcg.sh\n'
        script += 'python=$(which python3)\n'
        script += f'sipmgr={ch}\n'
        script += 'root_type="tile"\n'
        script += f'file_path="{input_dir}"\n'
        script += f'run_info="{run_info}"\n'
        script += f'run_number="{run_number}"\n'
        script += f'output_file="{output_dir}"\n'
        script += f'cat /proc/cpuinfo > {output_dir}/logs/cpuInfo/charge_fit_tile_ch{ch}_ov{ov}.log\n'
        script += '# Construct the input filename\n'
        script += f'ov={ov}\n'
        script += 'input_file="${file_path}/root/${run_info}_ov_${ov}.00_sipmgr_$(printf "%02d" $sipmgr)_${root_type}.root"\n'
        script += '# Construct and execute the command\n'
        script += 'charge_fit_command="$python charge_fit.py ${input_file} signal sigQ ${output_file}'+f' {parrent_path}/test/signal-fit-harvests/csv/run_{str(run_number).zfill(4)}.csv"\n'
        script += 'echo "Executing command: ${charge_fit_command}"\n'
        script += '$charge_fit_command\n'
        script += '\n'
        script += 'cd -\n'
        with open(f'{output_dir}/jobs/charge_fit_tile_ch{ch}_ov{ov}.sh','w') as file_tmp:
                file_tmp.write(script)

batch_script = '''
#!/bin/bash

# get procid from command line
procid=$1

# There are 6 ov values per channel, and 16 channels in total.
channel=$(( procid / 6 + 1 ))
override=$(( procid % 6 + 1 ))

# format override to have leading zeros if necessary
formatted_override=$(printf "%01d" $override)

# construct the script name based on the channel and override
'''
batch_script += "\n"
batch_script += 'script_name="'+f'{output_dir}/jobs/'+'charge_fit_tile_ch${channel}_ov${formatted_override}.sh"\n'

# run the real job script by the formatted file name
batch_script += 'bash "$script_name"\n'

with open(f'{output_dir}/big-submission.sh','w') as file_tmp:
        file_tmp.write(batch_script)
os.system(f"chmod +x {output_dir}/jobs/*.sh")
os.system(f"chmod +x {output_dir}/*.sh")
