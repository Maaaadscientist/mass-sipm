import os, sys
import subprocess
import glob
import yaml

# Execute a Python script

def list_txt_files(directory):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_files.append(filename)
    return txt_files

if len(sys.argv) < 4:
    raise OSError("Usage: python script/prepare_all_jobs.py <input_table_yaml> <output_dir> <analysis_type>")
else:
    input_tmp = sys.argv[1]
    output_tmp = sys.argv[2]    
    analysis_type = sys.argv[3]
table_path = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)
if analysis_type == "main": 
    runType = "main"
    binary_path = os.path.abspath("bin/skim-signal")
    file_type = "root"
elif analysis_type == "light":
    runType = "light"
    file_type = "root"
    binary_path = os.path.abspath("bin/skim-signal")
elif analysis_type == "dcr":
    runType = "main"
    file_type = "root"
    binary_path = os.path.abspath("bin/skim-dcr")
elif analysis_type =="signal-fit":
    runType = "main"
    file_type = "pdf"
elif analysis_type =="light-fit":
    runType = "light"
    file_type = "pdf"
elif analysis_type =="light-match":
    runType = "light"
    file_type = "root"
elif analysis_type =="light-match-bootstrap":
    runType = "light"
    file_type = "png"
elif analysis_type =="decoder":
    runType = "light"
    file_type = "csv"
elif analysis_type =="mainrun-light-fit":
    runType = "main"
    file_type = "pdf"
elif analysis_type == "dcr-fit":
    runType = "main"
    file_type = "csv"
elif analysis_type == "vbd":
    runType = "main"
    file_type = "pdf"
elif analysis_type == "harvest":
    runType = "main"
    file_type = "root"
main_runs = []
light_runs = []

if file_type == "root":
    threshold = 7000
    if analysis_type == "harvest":
        threshold = 5
elif file_type == "pdf":
    threshold = 1000
elif file_type == "png":
    threshold = 100
elif file_type == "csv":
    threshold = 1000

with open(table_path, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)   
    light_runs = yaml_data["light_runs"]
    main_runs = yaml_data["main_runs"]

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
    # Prompt the user for input
    choice2 = input("Interactive mode? (Y/A/R/N), Y: interactive mode, A: non-interactive mode and submit all, R: just resubmit failed jobs, N or others: exit): ").upper()
    
    # Check if the user wants to resubmit jobs
    if choice2 == "Y":
        interactive = "true"
        resubmit = "true"
    elif choice2 == "A":
        interactive = "false"
        resubmit = "false"
    elif choice2 == "R":
        interactive = "false"
        resubmit = "true"
    else:
        sys.exit("do nothing and exit")
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
        if analysis_type == "light-match-bootstrap":
            command = f'./script/check_jobs_multi-thread.sh {threshold} {output_dir}/{analysis_type}/{name_short} {interactive} {file_type} {resubmit}'
        else:
            command = f'./check_jobs.sh {threshold} {output_dir}/{analysis_type}/{name_short} {interactive} {file_type} {resubmit}'
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

        if not return_code == 0:
            print(f"Error executing script. Return code: {return_code}")
            print("Error output:")
            print(error.decode('utf-8'))
else:
    print("Jobs not resubmitted.")
