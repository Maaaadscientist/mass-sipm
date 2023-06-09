import os,sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python prepare_jobs <input_dir>")
else:
   input_tmp = sys.argv[1]
input_dir = os.path.abspath(input_tmp)

csv_dir = input_dir + "/plots"
all_data = []

for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_dir, filename)
        data = pd.read_csv(file_path)
        all_data.append(data)

combined_data = pd.concat(all_data, ignore_index=True)

combined_data.to_csv(f'{input_dir}/fit_results.csv', index=False)
