import os
import sys

with open("/workfs2/juno/wanghanwen/sipm-massive/config/tsn_analyzed.txt", "r") as file1:
    lines = file1.readlines()


script  = ''
script += '#!/bin/bash\n'
script += 'directory="/junofs/users/wanghanwen/plot_tiles/csv"\n'
script += 'if [ ! -d "$directory" ]; then\n'
script += '  echo "Directory does not exist. Creating directory..."\n'
script += '  mkdir -p "$directory"\n'
script += '  echo "Directory created."\n'
script += 'else\n'
script += '  echo "Directory already exists."\n'
script += 'fi\n'
script += 'source /workfs2/juno/wanghanwen/sipm-massive/env_lcg.sh\n'
script += 'PYTHON=$(which python3)\n'
script += 'cd $directory\n'

job_dir = "/junofs/users/wanghanwen/plot_tiles/jobs"
if not os.path.isdir(job_dir):
    os.makedirs(job_dir)

for aline in lines:
    tsn = int(aline.strip())
    with open(f"{job_dir}/parameter_{tsn}.sh", "w") as file1:
        job_content = f"$PYTHON /workfs2/juno/wanghanwen/sipm-massive/test/plot_all_results.py {tsn}\n"
        file1.write(script)
        file1.write(job_content)
os.system("chmod a+x /junofs/users/wanghanwen/plot_tiles/jobs/*.sh")
