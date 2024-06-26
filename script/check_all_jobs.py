import os, sys
import re

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
    file_type = "csv"
elif analysis_type =="signal-refit":
    runType = "main"
    file_type = "root"
elif analysis_type =="light-fit":
    runType = "light"
    file_type = "pdf"
elif analysis_type =="light-match":
    runType = "light"
    file_type = "csv"
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
elif analysis_type == "new-dcr":
    runType = "main"
    file_type = "csv"
elif analysis_type == "vbd":
    runType = "main"
    file_type = "csv"
elif analysis_type == "harvest":
    runType = "main"
    file_type = "root"
elif analysis_type == "main-reff":
    runType = "main"
    file_type = "root"
elif analysis_type == "main-match":
    runType = "main"
    file_type = "csv"
main_runs = []
light_runs = []


if os.path.isfile(table_path):
    with open(table_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)   
        light_runs = yaml_data["light_runs"]
        main_runs = yaml_data["main_runs"]
else:
    print(f"User defined single run number: {input_tmp}")
    
    def separate_string_and_number(input_string):
        # Using regular expression to split alphabetic and numeric parts
        match = re.match(r"([a-zA-Z]+)_(\d+)", input_string)
        
        if match:
            # Extracting string and number parts
            string_part = match.group(1)
            number_part = int(match.group(2))
            return string_part, number_part
        else:
            # If the input string doesn't match the pattern
            return None, None
    string_part, number_part = separate_string_and_number(input_tmp)
    if string_part is not None and number_part is not None:
        print("Run Type:", string_part)
        print("Number:", number_part)
        if string_part == "main":
            main_runs.append(number_part)
        else:
            light_runs.append(number_part)
    else:
        print("Invalid input string format.")
    #for line in file:
    #    line = line.strip()
    #    if line:
    #        key, value = line.split(':')
    #        main_runs.append(int(key.strip()))
    #        light_runs.append(int(value.strip()))

if analysis_type == "main-reff":
    check_type = "csv"
    file_type = "root"
elif analysis_type == "signal-fit" or analysis_type == "signal-refit":
    check_type = "csv"
    file_type = "csv"
else:
    check_type = file_type

if file_type == "root":
    threshold = 7000
    if analysis_type == "main":
        threshold = 2000000
    if analysis_type == "harvest":
        threshold = 5
    elif analysis_type == "main-reff":
        threshold = 1
    elif analysis_type == "main-match":
        threshold = 4000

elif file_type == "pdf":
    threshold = 1000
elif file_type == "png":
    threshold = 100
elif file_type == "csv":
    threshold = 100
    if analysis_type == "vbd":
        threshold == 20* 1024

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
    if resubmit == "true":
        choice3 = input("Keep jobs logs? (Y/N)): ").upper()
        if choice3 == "Y":
            keep_logs = "true"
        elif choice3 == "N":
            keep_logs = "false"
    else:
        keep_logs = "false"
    
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
    failed_jobs = ""
    failed_logs = ""
    for aFile in grouped_list:
        name_short = aFile.split("/")[-1].replace(".txt", "")
        run = int(name_short.split("_")[-1])
        run_type = name_short.split('_')[0]
        if run_type != runType:
            continue
        if run_type == "main" and not (int(run) in main_runs):
            continue
        if run_type == "light" and not (run in light_runs):
            continue
        if run_type != "main" and run_type != "light":
            continue
        print(name_short) 
        # Define the command to execute the shell script
        if interactive == "false" and resubmit == "false" and os.path.isfile(f"{output_dir}/{analysis_type}/{name_short}/big-submission.sh"):
            number_of_jobs = int(len(os.listdir(f"{output_dir}/{analysis_type}/{name_short}/jobs")))
            os.system(f'hep_sub {output_dir}/{analysis_type}/{name_short}/big-submission.sh ''-argu "%{ProcId}"'f' -n {number_of_jobs} -o /dev/null -e /dev/null')
        else:
            command = f'./script/check_jobs.sh {threshold} {output_dir}/{analysis_type}/{name_short} {interactive} {check_type} {file_type} {resubmit} {keep_logs}'
        #print(command)

        # Execute the shell script and capture the output interactively
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        #output, error = process.communicate()
        # Read and print the output line by line
            for line in process.stdout:
                print(line, end='')
                if "failed job path:" in line:
                    failed_jobs += line.split("failed job path:")[1].strip()
                    failed_jobs += "\n"
            if keep_logs:
                failed_logs = failed_jobs.replace("jobs", "logs")
            
            # Check the return code of the process
            return_code = process.wait()

            if not return_code == 0:
                print(f"Error executing script. Return code: {return_code}")
                print("Error output:")
                print(error.decode('utf-8'))
    with open(f"failed_jobs_{analysis_type}.txt", "w") as aFile:
        aFile.write(failed_jobs)
        aFile.close()
    with open(f"failed_logs_{analysis_type}.txt", "w") as aFile:
        aFile.write(failed_logs)
        aFile.close()
else:
    print("Jobs not resubmitted.")
