import os, sys
import subprocess
import glob

# Execute a Python script

def list_txt_files(directory):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_files.append(filename)
    return txt_files

if len(sys.argv) < 3:
    raise OSError("Usage: python script/prepare_all_jobs.py <table_path> <input_path> <output_path> ")
else:
    table_tmp = sys.argv[1]    
    input_tmp = sys.argv[2]
    output_tmp = sys.argv[3]    

table_path = os.path.abspath(table_tmp)
input_dir = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
runType = "main"

    
if not os.path.isdir(f'{output_dir}'):
    os.makedirs(f'{output_dir}')
main_runs = []
light_runs = []

with open(table_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            main_runs.append(int(key.strip()))
            light_runs.append(int(value.strip()))

#analysis_list = ['dcr-fit', 'signal-fit', 'vbd', 'harvest', 'light-fit']
analysis_list = [ 'harvest', ]


# Remove elements that don't contain "main" or "light"
filtered_list = [item for item in list_txt_files("datasets") if "main" in item or "light" in item]
# Sort the list in ascending order
sorted_list = sorted(filtered_list, key=lambda x: int(x.split('_')[2][:-4]))
# Group the "light_run" elements together while preserving the original order
grouped_list = sorted(sorted_list, key=lambda x: ("light_run" not in x, sorted_list.index(x)))

for aFile in grouped_list:
    name_short = aFile.split("/")[-1].replace(".txt", "")
    run = int(name_short.split("_")[-1])
    run_type = name_short.split('_')[0]
    if run_type != runType:
        continue
    if run_type == "main" and not (run in main_runs):
        continue
    if run_type == "light" and not (run in light_runs):
        continue
    if run_type != "main" and run_type != "light":
        continue
    print(name_short)

    for key in analysis_list:
        filepath = output_dir + "/" + key + "/" + name_short
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        command1 = f'cp  {input_dir}/{key}/{name_short}/pdf/*.pdf  {filepath}'
        command2 = f'cp  {input_dir}/{key}/{name_short}/root/*.root  {filepath}'
        process = subprocess.Popen(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        process = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Execute the shell script
    #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
