#!/usr/bin/env python3
import os, sys

if len(sys.argv) < 3:
    print("Usage: python prepare_dcr_jobs.py <input_dir> <csv_dir> <output_dir>")
else:
   input_tmp = sys.argv[1]
   csv_tmp = sys.argv[2]
   output_tmp = sys.argv[3]
input_dir = os.path.abspath(input_tmp)
csv_dir = os.path.abspath(csv_tmp)
output_dir = os.path.abspath(output_tmp)
run_info = input_dir.split("/")[-1]
run_number = int(run_info.split("_")[-1])

if not os.path.isdir(output_dir + "/jobs"):
    os.makedirs(output_dir + "/jobs")

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
        script += 'cp /workfs2/juno/wanghanwen/sipm-massive/script/dcr_fit.py .\n'
        script += 'cp /workfs2/juno/wanghanwen/sipm-massive/env_lcg.sh .\n'
        script += '. ./env_lcg.sh\n'
        script += 'python=$(which python3)\n'
        script += f'sipmgr={ch}\n'
        script += 'root_type="tile"\n'
        script += f'file_path="{input_dir}"\n'
        script += f'run_info="{run_info}"\n'
        script += f'run_number="{run_number}"\n'
        script += f'output_file="{output_dir}"\n'
        script += f'csv_dir="{csv_dir}"\n'
        script += '# Construct the input filename\n'
        script += f'ov={ov}\n'
        script += 'input_file="${file_path}/root/${run_info}_ov_${ov}.00_sipmgr_$(printf "%02d" $sipmgr)_${root_type}.root"\n'
        script += '# Construct and execute the command\n'
        script += 'charge_fit_command="$python dcr_fit.py ${input_file} dcr dcrQ ${csv_dir}/csv ${output_file}"\n'
        script += 'echo "Executing command: ${charge_fit_command}"\n'
        script += '$charge_fit_command\n'
        script += '\n'
        script += 'cd -\n'
        with open(f'{output_dir}/jobs/dcr_fit_tile_ch{ch}_ov{ov}.sh','w') as file_tmp:
                file_tmp.write(script)

os.system(f"chmod +x {output_dir}/jobs/*.sh")
