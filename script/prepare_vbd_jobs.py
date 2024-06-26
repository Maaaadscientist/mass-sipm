#!/usr/bin/env python3
import os, sys

if len(sys.argv) < 3:
    print("Usage: python prepare_vbd_jobs.py <input_dir> <output_dir> <binary_path>")
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
if not os.path.isdir(output_dir + "/pdf"):
    os.makedirs(output_dir + "/pdf")
if not os.path.isdir(output_dir + "/csv"):
    os.makedirs(output_dir + "/csv")

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
script += f'cp {parrent_path}/script/robust_vbd.py .\n'
script += f'cp {parrent_path}/test/simple_merge_csv.py .\n'
script += f'cp {parrent_path}/env_lcg.sh .\n'
script += '. ./env_lcg.sh\n'
script += 'python=$(which python3)\n'
script += f'file_path="{input_dir}"\n'
script += f'run_info="{run_info}"\n'
script += f'run_number="{run_number}"\n'
script += f'output_file="{output_dir}"\n'
script += '# Construct the input filename\n'
script += 'input_file="${file_path}/csv"\n'
script += '# Construct and execute the command\n'
script += 'charge_fit_command1="$python simple_merge_csv.py ${input_file} tmp.csv"\n'
script += 'charge_fit_command2="$python robust_vbd.py tmp.csv ${output_file}"\n'
script += 'echo "Executing command: ${charge_fit_command1}"\n'
script += '$charge_fit_command1\n'
script += 'echo "Executing command: ${charge_fit_command2}"\n'
script += '$charge_fit_command2\n'
script += '\n'
script += 'cd -\n'
with open(f'{output_dir}/jobs/get_vbd_run{run_number}.sh','w') as file_tmp:
        file_tmp.write(script)

os.system(f"chmod +x {output_dir}/jobs/*.sh")
