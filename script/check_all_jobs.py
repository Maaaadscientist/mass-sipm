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
    raise OSError("Usage: python script/prepare_all_jobs.py <input_table_yaml> <output_dir> <run_type>")
else:
    input_tmp = sys.argv[1]
    output_tmp = sys.argv[2]    
if len(sys.argv) == 4:
    runType = sys.argv[3]
else:
    runType = "main"
table_path = os.path.abspath(input_tmp)
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

# Remove elements that don't contain "main" or "light"
filtered_list = [item for item in list_txt_files("datasets") if "main" in item or "light" in item]
# Sort the list in ascending order
sorted_list = sorted(filtered_list, key=lambda x: int(x.split('_')[2][:-4]))
# Group the "light_run" elements together while preserving the original order
grouped_list = sorted(sorted_list, key=lambda x: ("light_run" not in x, sorted_list.index(x)))

# Prompt the user for input
choice = input("Ready to resubmit jobs? (Y/N): ").upper()

# Check if the user wants to resubmit jobs
if choice == "Y":
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
       
        # Define the command to execute the shell script
        command = f'./check_jobs.sh 1024 {output_dir}/{name_short} 1'
        #print(command)

        # Execute the shell script
        #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        # Execute the shell script and capture the output interactively
        #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        #output, error = process.communicate()
        # Read and print the output line by line
        for line in process.stdout:
            print(line, end='')
        
        # Check the return code of the process
        return_code = process.wait()

        if return_code == 0:
            print(f"Script executed successfully for {run_type} run {run}")
        else:
            print(f"Error executing script. Return code: {return_code}")
            print("Error output:")
            print(error.decode('utf-8'))
else:
    print("Jobs not resubmitted.")
