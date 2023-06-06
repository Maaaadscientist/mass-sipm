#!/usr/bin/env python3
import os, sys
from copy import deepcopy
if len(sys.argv) < 3:
    print("Usage: python prepare_jobs <input_file> <output_dir>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
file_list =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
eos_mgm_url = "root://junoeos01.ihep.ac.cn"
directory = "/tmp/tao-sipmtest"
output_dir = os.path.abspath(output_tmp)
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

for file_path in files:
    if "path=" in file_path:
        continue 
    if not ".data" in file_path:
        continue
    file_name = file_path.split("/")[-1]
    print(file_name)
    components = file_name.split("_")
    run_type = components[0]
    if run_type == "main":
        run_number = components[2]
        ov = int(float(components[4]))
        channel = int(components[6])
        sipm_type = components[7]
        print(run_type, run_number, ov, channel, sipm_type)
        script_tmp = deepcopy(script)
        script_tmp += f'/usr/bin/eos cp {file_path} $directory\n'
        script_tmp += 'cd $directory\n'
        script_tmp += 'sleep 3\n'
        script_tmp += 'cp /junofs/users/wanghanwen/sipm-massive/test.yaml .\n'
        script_tmp += 'cp /junofs/users/wanghanwen/sipm-massive/env_lcg.sh .\n'
        script_tmp += '. ./env_lcg.sh\n'
        output_name = "_".join(components[0:-3])
        script_tmp += f'/junofs/users/wanghanwen/sipm-massive/bin/scan -i {file_name} -c test.yaml -r {run_number} -v {ov} -t {run_type}_{sipm_type}_ch{channel} -o {output_dir}/{output_name}.root\n'
        script_tmp += 'sleep 5\n'
        script_tmp += f'rm -f {file_name}\n'
        script_tmp += 'cd -\n'
        with open(f'{output_dir}/jobs/{output_name}.sh','w') as file_tmp:
            file_tmp.write(script_tmp)
    elif run_type == "light":
        run_number = components[2]
        channel = int(components[4])
        sipm_type = components[5]
        print(run_type, run_number, channel, sipm_type)
        script_tmp = deepcopy(script)
        script_tmp += f'/usr/bin/eos cp {file_path} $directory\n'
        script_tmp += 'cd $directory\n'
        script_tmp += 'sleep 3\n'
        script_tmp += 'cp /junofs/users/wanghanwen/sipm-massive/test.yaml .\n'
        script_tmp += 'cp /junofs/users/wanghanwen/sipm-massive/env_lcg.sh .\n'
        script_tmp += '. ./env_lcg.sh\n'
        output_name = "_".join(components[0:-3])
        script_tmp += f'/junofs/users/wanghanwen/sipm-massive/bin/scan -i {file_name} -c test.yaml -r {run_number} -v 0 -t {run_type}_{sipm_type}_ch{channel} -o {output_dir}/{output_name}.root\n'
        script_tmp += 'sleep 5\n'
        script_tmp += f'rm -f {file_name}\n'
        script_tmp += 'cd -\n'
        with open(f'{output_dir}/jobs/{output_name}.sh','w') as file_tmp:
            file_tmp.write(script_tmp)
       

os.system(f"chmod +x {output_dir}/jobs/*.sh")
os.system(f"cp submit_jobs.sh {output_dir}")
