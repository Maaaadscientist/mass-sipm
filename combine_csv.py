import os
import pandas as pd

directory = '/path/to/csv/files'
all_data = []

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        data = pd.read_csv(file_path)
        all_data.append(data)

combined_data = pd.concat(all_data, ignore_index=True)

