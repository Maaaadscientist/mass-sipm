#!/usr/bin/env python3
import os, sys

if len(sys.argv) < 4:
    print("Usage: python prepare_light_jobs.py <input_dir> <output_dir>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
   binary_path = sys.argv[3]
input_dir = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
parrent_path = "/".join(binary_path.split("/")[0:-2])
run_info = input_dir.split("/")[-1]
run_number = int(run_info.split("_")[-1])

if not os.path.isdir(output_dir + "/jobs"):
    os.makedirs(output_dir + "/jobs")

#for point in range(1, 65):

for filename in os.listdir(f"{input_dir}/root"):
    filename_short = filename.replace(".root", "")
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
    script += f'cp {parrent_path}/script/light_fit.py .\n'
    script += f'cp {parrent_path}/env_lcg.sh .\n'
    script += '. ./env_lcg.sh\n'
    script += 'python=$(which python3)\n'
    script += f'file_path="{input_dir}"\n'
    script += f'output_file="{output_dir}"\n'
    script += '# Construct the input filename\n'
    script += 'input_file="${file_path}/root/' + f'{filename}"\n'
    script += '# Construct and execute the command\n'
    script += 'charge_fit_command="$python light_fit.py ${input_file} signal sigQ ${output_file}"\n'
    script += 'echo "Executing command: ${charge_fit_command}"\n'
    script += '$charge_fit_command\n'
    script += '\n'
    script += 'cd -\n'
    with open(f'{output_dir}/jobs/{filename_short}.sh','w') as file_tmp:
            file_tmp.write(script)

os.system(f"chmod +x {output_dir}/jobs/*.sh")
