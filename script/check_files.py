import os,sys
import csv

def check_missing_files(directory,run, var_type):
    missing_files = []
    peak_num = [2, 2, 2, 3, 3, 4, 4]
    for sig_type in var_type:
        for ov in range(1, 7):
            for tile in range(1, 17):
                for po in range(16):
                    filename = f"{sig_type}_main_run{run}_ov{ov}_tile_ch{tile}_po{po}.csv"
                    filepath = os.path.join(directory, filename)
                    if not os.path.isfile(filepath):
                        missing_files.append(filename)
                    else:
                        with open(filepath, 'r') as file:
                            reader = csv.reader(file)
                            rows = list(reader)
                            if len(rows) < peak_num[ov] + 1:
                                missing_files.append(filename + f" :{ov}V less than {peak_num[ov]} peaks")

    return missing_files

# Specify your directory path here
directory_path = sys.argv[1]
run = sys.argv[2]
input_dir = os.path.abspath(directory_path)
var_type = sys.argv[3:]

missing_files = check_missing_files(input_dir, run, var_type)

if len(missing_files) == 0:
    print("All files are present in the directory.")
else:
    print("Missing files:")
    for file in missing_files:
        print(file)

