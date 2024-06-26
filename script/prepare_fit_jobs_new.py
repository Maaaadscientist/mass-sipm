#!/usr/bin/env python3
import os, sys

if len(sys.argv) < 3:
    print("Usage: python prepare_fit_jobs_new.py <input_dir> <output_dir>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
input_dir = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
run_info = input_dir.split("/")[-1]
run_number = int(run_info.split("_")[-1])

if not os.path.isdir(output_dir + "/signal-jobs"):
    os.makedirs(output_dir + "/signal-jobs")
template = """for ov in {1..6}; do
  for ch in {0..15}; do
    # Construct the input filename
    input_file="${file_path}/${run_info}_ov_${ov}.00_sipmgr_$(printf "%02d" $sipmgr)_${root_type}.root"

    # Construct and execute the command
    charge_fit_command="$python fit.py ${input_file} signal sigQ_ch${ch} ${output_file}"
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
    script += 'cp /junofs/users/wanghanwen/sipm-massive/script/fit.py .\n'
    script += 'cp /junofs/users/wanghanwen/sipm-massive/env_lcg.sh .\n'
    script += '. ./env_lcg.sh\n'
    script += 'python=$(which python3)\n'
    script += f'sipmgr={ch}\n'
    script += 'root_type="tile"\n'
    script += f'file_path="{input_dir}"\n'
    script += f'run_info="{run_info}"\n'
    script += f'run_number="{run_number}"\n'
    script += f'output_file="{output_dir}"\n'
    script += template
    script += '\n'
    script += 'cd -\n'
    with open(f'{output_dir}/signal-jobs/fit_tile{ch}.sh','w') as file_tmp:
            file_tmp.write(script)

os.system(f"chmod +x {output_dir}/signal-jobs/*.sh")
os.system(f"cp submit_jobs.sh {output_dir}")
os.system(f"chmod +x {output_dir}/submit_jobs.sh ")
os.system(f"cp script/combine_csv.py {output_dir}")
