#!/usr/bin/env python3
import os, sys
from copy import deepcopy
import yaml
if len(sys.argv) < 4:
    print("Usage: python prepare_jobs <input_file> <output_dir> <executable_path>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
   binary_tmp = sys.argv[3]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
isDCR = 'dcr' in binary_tmp
file_list =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
eos_mgm_url = "root://junoeos01.ihep.ac.cn"
directory = "/tmp/tao-sipmtest"
input_file = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
binary_path = os.path.abspath(binary_tmp)
parrent_path = "/".join(binary_path.split("/")[0:-2])
name_short = input_file.split("/")[-1].replace(".txt", "")
comps = name_short.split("_")
runNumber = comps[2]
output_dir += "/" + name_short
#output_dir = "/junofs/users/wanghanwen/main_run_0075"

script  = ''
script += '#!/bin/bash\n'
script += 'export EOS_MGM_URL="root://junoeos01.ihep.ac.cn"\n'
script += 'directory="/tmp/tao-sipmtest"\n'
script += 'if [ ! -d "$directory" ]; then\n'
script += '  echo "Directory does not exist. Creating directory..."\n'
script += '  mkdir -p "$directory"\n'
script += '  echo "Directory created."\n'
script += 'else\n'
script += '  echo "Directory already exists."\n'
script += 'fi\n'


with open(file_list, "r") as f:
    files = f.read().splitlines()

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
if not os.path.isdir(output_dir + "/jobs"):
    os.mkdir(output_dir + "/jobs")
if not os.path.isdir(output_dir + "/csv"):
    os.mkdir(output_dir + "/csv")

count = 0
for file_path in files:
    if "path=" in file_path:
        continue 
    if not ".log" in file_path:
        continue
    file_name = file_path.split("/")[-1]
    count += 1
    file_name_short = file_name.replace(".log", "")
    components = file_name_short.split("_")
    if components[0] == "lfrun":
        run_number = int(components[1])
    elif components[0] == "light": 
        run_number = int(components[2])
    else:
        raise ValueError("Unrecognized log name!")
    script_tmp = deepcopy(script)
    script_tmp += f'/usr/bin/eos cp {file_path} $directory\n'
    script_tmp += 'cd $directory\n'
    script_tmp += 'sleep 3\n'
    script_tmp += f'cp {parrent_path}/env_lcg.sh .\n'
    script_tmp += f'cp {binary_path} .\n'
    script_tmp += '. ./env_lcg.sh\n'
    script_tmp += 'python=$(which python3)\n'
    script_tmp += f'input_file={file_name}\n'
    script_tmp += f'output_file={output_dir}\n'
    #output_name = "_".join(components[0:-3])
    script_tmp += f'decoder_command="$python {binary_path}' + ' ${input_file}  ${output_file}/csv"\n'
    script_tmp += 'echo "Executing command: ${decoder_command}"\n'
    script_tmp += '$decoder_command\n'
    script_tmp += 'sleep 5\n'
    script_tmp += f'rm -f {file_name}\n'
    script_tmp += 'cd -\n'
    with open(f'{output_dir}/jobs/decoder_run{run_number}.sh','w') as file_tmp:
        file_tmp.write(script_tmp)

if count == 0:
    print(f"No log file found for Run-{runNumber}")
else:
    os.system(f"chmod +x {output_dir}/jobs/*.sh")
