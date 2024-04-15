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
    raise OSError("Usage: python script/prepare_all_jobs.py <input_table> <output_path> <analysis_type>")
else:
    input_tmp = sys.argv[1]
    analysis_type = sys.argv[3]
    output_tmp = sys.argv[2]    

table_path = os.path.abspath(input_tmp)
output_dir = os.path.abspath(output_tmp)

if analysis_type == "main": 
    runType = "main"
    binary_path = os.path.abspath("bin/skim-signal")
elif analysis_type == "light":
    runType = "light"
    binary_path = os.path.abspath("bin/skim-signal")
elif analysis_type == "light-match":
    runType = "light"
    binary_path = os.path.abspath("../light-match/bin/light_match")
elif analysis_type == "main-match":
    runType = "main"
    binary_path = os.path.abspath("../light-match/bin/light_match")
elif analysis_type == "decoder":
    runType = "light"
    binary_path = os.path.abspath("script/extract_decoder.py")
elif analysis_type == "light-match-bootstrap":
    runType = "light"
    binary_path = os.path.abspath("../light-match/bin/light_match")
elif analysis_type == "dcr":
    runType = "main"
    binary_path = os.path.abspath("bin/skim-dcr")
elif analysis_type == "new-dcr":
    runType = "main"
    binary_path = os.path.abspath("bin/new_dcr")
elif analysis_type =="signal-fit":
    runType = "main"
    binary_path = os.path.abspath("script/charge_fit.py")
elif analysis_type =="signal-refit":
    runType = "main"
    binary_path = os.path.abspath("script/charge_fit.py")
elif analysis_type == "light-fit":
    runType = "light"
    binary_path = os.path.abspath("script/light_fit.py")
elif analysis_type == "mainrun-light-fit":
    runType = "main"
elif analysis_type == "dcr-fit":
    runType = "main"
elif analysis_type == "main-reff":
    runType = "main"
    binary_path = os.path.abspath("../light-match/bin/main_reff")
elif analysis_type == "vbd":
    runType = "main"
    binary_path = os.path.abspath("script/get_vbd_new.py")
elif analysis_type == "harvest":
    runType = "main"
    
#print(analysis_type, analysis_type == "signal-refit")
if not os.path.isdir(f'{output_dir}/{analysis_type}'):
    if not (analysis_type == "signal-refit"):
        os.makedirs(f'{output_dir}/{analysis_type}')
        #print("test")
main_runs = []
light_runs = []

print(table_path)
if os.path.isfile(table_path):
    with open(table_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)   
        print(yaml_data)
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

print(main_runs)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
output_dir = os.path.abspath(output_tmp)

#if analysis_type == "main" or analysis_type == "light":
#    os.system("sh get_datasets.sh")
# Get the current directory
current_directory = os.getcwd()


# Iterate over the found files and remove them
if analysis_type == "main" or analysis_type == "light":
    # Find all "*{runType}.log" files in the current directory
    log_files = glob.glob(os.path.join(current_directory, f'*{runType}.log'))
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
    if analysis_type == "signal-fit":
        subprocess.run(['python', 'script/prepare_signal_jobs.py', f'{output_dir}/main/{name_short}', f'{output_dir}/{analysis_type}/{name_short}', f'{binary_path}'])
    elif analysis_type == "signal-refit":
        subprocess.run(['python', 'script/prepare_refit_jobs.py', f'{output_dir}/main/{name_short}', f'{output_dir}/signal-refit/{name_short}', f'{binary_path}'])
    elif analysis_type == "dcr-fit":
        subprocess.run(['python', 'script/prepare_dcr_jobs.py', f'{output_dir}/dcr/{name_short}',f'{output_dir}/signal-fit/{name_short}', f'{output_dir}/{analysis_type}/{name_short}'])
    elif analysis_type == "light-fit":
        subprocess.run(['python', 'script/prepare_light_jobs.py', f'{output_dir}/light/{name_short}', f'{output_dir}/{analysis_type}/{name_short}', f'{binary_path}'])
    elif analysis_type == "mainrun-light-fit":
        subprocess.run(['python', 'script/prepare_mainrun_light_jobs.py', f'{output_dir}/main/{name_short}', f'{output_dir}/{analysis_type}/{name_short}'])
    elif analysis_type == "vbd":
        subprocess.run(['python', 'script/prepare_vbd_jobs.py', f'{output_dir}/signal-fit/{name_short}', f'{output_dir}/{analysis_type}/{name_short}', f'{binary_path}'])
    elif analysis_type == "harvest":
        subprocess.run(['python', 'script/prepare_harvest_jobs.py', f'{output_dir}', f'{name_short}', f'{output_dir}/{analysis_type}/{name_short}'])
    elif analysis_type == "main" or analysis_type == "light" or analysis_type == "dcr":
        subprocess.run(['python', 'script/prepare_skim_jobs.py',f'datasets/{aFile}', f'{output_dir}/{analysis_type}', f'{binary_path}'])
    elif analysis_type == 'new-dcr':
        subprocess.run(['python', 'script/prepare_new_dcr_jobs.py',f'datasets/{aFile}', f'{output_dir}/{analysis_type}', f'{binary_path}', f'{output_dir}/signal-refit'])
        #subprocess.run(['python', 'script/prepare_new_dcr_jobs.py',f'datasets/{aFile}', f'{output_dir}/{analysis_type}', f'{binary_path}', f'{output_dir}/signal-fit'])
    elif analysis_type == "light-match":
        subprocess.run(['python', 'script/prepare_match_jobs.py',f'datasets/{aFile}', f'{output_dir}/{analysis_type}', f'{binary_path}'])
    elif analysis_type == "main-match":
        subprocess.run(['python', 'script/prepare_main_match_jobs.py',f'{output_dir}/main-reff/{name_short}', f'{output_dir}/{analysis_type}', f'{binary_path}'])
    elif analysis_type == "light-match-bootstrap":
        subprocess.run(['python', 'script/prepare_bootstrap_jobs.py',f'datasets/{aFile}', f'{output_dir}/{analysis_type}', f'{binary_path}'])
    elif analysis_type == "decoder":
        subprocess.run(['python', 'script/prepare_decoder_jobs.py',f'datasets/{aFile}', f'{output_dir}/{analysis_type}', f'{binary_path}'])
    elif analysis_type == "main-reff":
        subprocess.run(['python', 'script/prepare_match_jobs.py',f'datasets/{aFile}', f'{output_dir}/{analysis_type}', f'{binary_path}'])
