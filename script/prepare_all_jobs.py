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

if len(sys.argv) < 4:
    raise OSError("Usage: python script/prepare_all_jobs.py <input_table> <output_dir> <binary_path> <run_type>")
else:
    input_tmp = sys.argv[1]
    output_tmp = sys.argv[2]    
    binary_tmp = sys.argv[3]
if len(sys.argv) == 5:
    runType = sys.argv[4]
else:
    runType = "main"
table_path = os.path.abspath(input_tmp)
binary_path = os.path.abspath(binary_tmp)
output_dir = os.path.abspath(output_tmp)
main_runs = []
light_runs = []

with open(table_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            main_runs.append(int(key.strip()))
            light_runs.append(int(value.strip()))

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
output_dir = os.path.abspath(output_tmp)

os.system("sh get_datasets.sh")
# Get the current directory
current_directory = os.getcwd()

# Find all "*{runType}.log" files in the current directory
log_files = glob.glob(os.path.join(current_directory, f'*{runType}.log'))

# Iterate over the found files and remove them
for file_path in log_files:
    os.remove(file_path)
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
    #os.system(f"python3 script/prepare_skim_jobs.py datasets/{aFile} {output_dir} {input_table}")
    subprocess.run(['python', 'script/prepare_skim_jobs.py',f'datasets/{aFile}', f'{output_dir}', f'{binary_path}'])

