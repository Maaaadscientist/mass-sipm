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
isMain = 'signal' in binary_tmp
file_list =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
eos_mgm_url = "root://junoeos01.ihep.ac.cn"
directory = "/tmp/tao-sipmtest"
input_file = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
binary_path = os.path.abspath(binary_tmp)
parrent_path = "/".join(binary_path.split("/")[0:-2])
name_short = input_file.split("/")[-1].replace(".txt", "")
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


general_run_number = 0
with open(file_list, "r") as f:
    files = f.read().splitlines()

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
if not os.path.isdir(output_dir + "/jobs"):
    os.mkdir(output_dir + "/jobs")
if not os.path.isdir(output_dir + "/root"):
    os.mkdir(output_dir + "/root")

count = 0
for file_path in files:
    file_name = file_path.split("/")[-1]
    components = file_name.split("_")
    run_type = components[0]
    if "path=" in file_path:
        continue 
    if not ".data" in file_path:
        continue
    if isDCR and 'reff' in file_name:
        continue
    if run_type == "main" and 'reff' in file_name:
        continue
    count += 1
    if run_type == "main":
        run_number = components[2]
        general_run_number = run_number
        ov = int(float(components[4]))
        channel = int(components[6])
        sipm_type = components[7]
        script_tmp = deepcopy(script)
        script_tmp += f'/usr/bin/eos cp {file_path} $directory\n'
        script_tmp += 'cd $directory\n'
        script_tmp += 'sleep 3\n'
        script_tmp += f'cp {parrent_path}/config/new.yaml .\n'
        script_tmp += f'cp {parrent_path}/env_lcg.sh .\n'
        script_tmp += '. ./env_lcg.sh\n'
        output_name = "_".join(components[0:-3])
        script_tmp += f'{binary_path} -i {file_name} -c new.yaml -r {run_number} -v {ov} -t {run_type}_{sipm_type}_ch{channel} -o {output_dir}/root/{output_name}.root\n'
        script_tmp += 'sleep 5\n'
        script_tmp += f'rm -f {file_name}\n'
        script_tmp += 'cd -\n'
        with open(f'{output_dir}/jobs/{output_name}.sh','w') as file_tmp:
            file_tmp.write(script_tmp)
    elif run_type == "light":
        run_number = components[2]
        channel = int(components[4])
        sipm_type = components[5]
        script_tmp = deepcopy(script)
        script_tmp += f'/usr/bin/eos cp {file_path} $directory\n'
        script_tmp += 'cd $directory\n'
        script_tmp += 'sleep 3\n'
        script_tmp += f'cp {parrent_path}/config/new.yaml .\n'
        script_tmp += f'cp {parrent_path}/env_lcg.sh .\n'
        script_tmp += '. ./env_lcg.sh\n'
        output_name = "_".join(components[0:-3])
        script_tmp += f'{binary_path} -i {file_name} -c new.yaml -r {run_number} -v 0 -t {run_type}_{sipm_type}_ch{channel} -o {output_dir}/root/{output_name}.root\n'
        script_tmp += 'sleep 5\n'
        script_tmp += f'rm -f {file_name}\n'
        script_tmp += 'cd -\n'
        with open(f'{output_dir}/jobs/{output_name}.sh','w') as file_tmp:
            file_tmp.write(script_tmp)
    elif run_type == "light-match":
        run_number = components[2]
        point_number = components[4]
        script_tmp = deepcopy(script)
        script_tmp += f'/usr/bin/eos cp {file_path} $directory\n'
        script_tmp += 'cd $directory\n'
        script_tmp += 'sleep 3\n'
        script_tmp += f'cp {parrent_path}/*.yaml .\n'
        script_tmp += f'cp {parrent_path}/data/* .\n'
        script_tmp += f'cp {parrent_path}/env_lcg.sh .\n'
        script_tmp += '. ./env_lcg.sh\n'
        output_name = "_".join(components[0:-3])
        script_tmp += f'{binary_path} -i {file_name} -c new.yaml -o {output_dir}/root\n'
        script_tmp += 'sleep 5\n'
        script_tmp += f'rm -f {file_name}\n'
        script_tmp += 'cd -\n'
        with open(f'{output_dir}/jobs/Run{run_number}_Point{point_number}.sh','w') as file_tmp:
            file_tmp.write(script_tmp)
if not isDCR:
    if "main" in name_short and count != 96:
        print(f"{name_short} : WARNING, there are only {count} lines and it's less than 96")
        with open(f"incompleteDataInfo-{name_short.split('_')[0]}.log", 'a') as file:
            line_of_text = f"main run {general_run_number} {count} 96"
            file.write(line_of_text + '\n')
    elif "light" in name_short and count != 64:
        print(f"{name_short} : WARNING, there are only {count} lines and it's less than 64")
        with open(f"incompleteDataInfo-{name_short.split('_')[0]}.log", 'a') as file:
            line_of_text = f"light run {general_run_number} {count} 64"
            file.write(line_of_text + '\n')
    else:
        print(f"{name_short} : Success, there are {count} lines in the datasets ({'64 for light' if name_short.split('_')[0] == 'light' else '96 for main'})")
        with open(f"completeDataInfo-{name_short.split('_')[0]}.log", 'a') as file:
            line_of_text = f"{name_short.split('_')[0]} run {general_run_number} {count} {64 if name_short.split('_')[0] == 'light' else 96}"
            file.write(line_of_text + '\n')
else:
    print(f"{name_short} DCR jobs are created")

os.system(f"chmod +x {output_dir}/jobs/*.sh")
