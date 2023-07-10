import os, sys

def list_txt_files(directory):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_files.append(filename)
    return txt_files

if len(sys.argv) < 2:
    raise OSError("Usage: python script/prepare_all_jobs.py <output_dir>")
else:
    output_tmp = sys.argv[1]    
output_dir = os.path.abspath(output_tmp)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
output_dir = os.path.abspath(output_tmp)
for aFile in list_txt_files("datasets"):
    os.system(f"python3 script/prepare_skim_jobs.py datasets/{aFile} {output_dir}")

