import os,sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python prepare_jobs <input_dir>")
else:
   input_tmp = sys.argv[1]
input_dir = os.path.abspath(input_tmp)

sub_dir = input_dir.split("/")[-1]

csv_dir = input_dir
all_data = []

output_dir = "/" + "/".join(input_dir.split("/")[1:-1])
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_dir, filename)
        data = pd.read_csv(file_path)
        all_data.append(data)

combined_data = pd.concat(all_data, ignore_index=True)

combined_data.to_csv(f'{output_dir}/fit_results_{sub_dir}.csv', index=False)
