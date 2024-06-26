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
map_path = "/workfs2/juno/wanghanwen/sipm-massive/test/map_outputs/root"
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


with open(file_list, "r") as f:
    files = f.read().splitlines()

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
if not os.path.isdir(output_dir + "/jobs"):
    os.mkdir(output_dir + "/jobs")
if not os.path.isdir(output_dir + "/csv"):
    os.mkdir(output_dir + "/csv")

if "main" in binary_path.split("/")[-1]:
    if not os.path.isdir(output_dir + "/root"):
        os.mkdir(output_dir + "/root")
 
count = 0
for file_path in files:
    if "path=" in file_path:
        continue 
    if not ".data" in file_path:
        continue
    file_name = file_path.split("/")[-1]
    if isDCR and 'reff' in file_name:
        continue
    if "tile" in file_name:
        continue
    count += 1
    components = file_name.split("_")
    file_name_short = file_name.replace(".data", "")
    run_number = int(components[2])
    script_tmp = deepcopy(script)
    script_tmp += f'/usr/bin/eos cp {file_path} $directory\n'
    script_tmp += 'cd $directory\n'
    script_tmp += 'sleep 3\n'
    script_tmp += f'cp {parrent_path}/test.yaml .\n'
    script_tmp += f'cp {parrent_path}/env_lcg.sh .\n'
    script_tmp += '. ./env_lcg.sh\n'
    #output_name = "_".join(components[0:-3])
    if "light" in file_path:
        script_tmp += f'{binary_path} -i {file_name} -c test.yaml -o {output_dir}/csv --map {map_path}/preciseMap_light_run_{run_number}.root\n'
    elif "main" in file_path:
        script_tmp += f'{binary_path} -i {file_name} -c test.yaml -o {output_dir}/csv --map {map_path}/preciseMap_main_run_{run_number}.root\n'
    script_tmp += 'sleep 5\n'
    script_tmp += f'rm -f {file_name}\n'
    script_tmp += 'cd -\n'
    if "light" in file_path:
        point_number = int(components[4])
        with open(f'{output_dir}/jobs/Run{run_number}_Point{point_number}.sh','w') as file_tmp:
            file_tmp.write(script_tmp)
    elif "main" in file_path:
        with open(f'{output_dir}/jobs/{file_name_short}.sh','w') as file_tmp:
            file_tmp.write(script_tmp)
if not isDCR:
    if "main" in name_short and count != 96:
        print(f"{name_short} : WARNING, there are only {count} lines and it's less than 192")
        with open(f"incompleteDataInfo-{name_short.split('_')[0]}.log", 'a') as file:
            line_of_text = f"main run {run_number} {count} 192"
            file.write(line_of_text + '\n')
    elif "light" in name_short and count != 64:
        print(f"{name_short} : WARNING, there are only {count} lines and it's less than 64")
        with open(f"incompleteDataInfo-{name_short.split('_')[0]}.log", 'a') as file:
            line_of_text = f"light run {run_number} {count} 64"
            file.write(line_of_text + '\n')
    else:
        print(f"{name_short} : Success, there are {count} lines in the datasets ({'64 for light' if name_short.split('_')[0] == 'light' else '192 for main'})")
        with open(f"completeDataInfo-{name_short.split('_')[0]}.log", 'a') as file:
            line_of_text = f"{name_short.split('_')[0]} run {run_number} {count} {64 if name_short.split('_')[0] == 'light' else 192}"
            file.write(line_of_text + '\n')
else:
    print(f"{name_short} DCR jobs are created")


os.system(f"chmod +x {output_dir}/jobs/*.sh")
