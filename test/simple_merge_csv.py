import os
import sys
import pandas as pd


def read_csv_file(csv_file):
    if os.path.isdir(csv_file):
        all_data = []
        for filename in os.listdir(csv_file):
            if filename.endswith(".csv"):
                file_path = os.path.join(csv_file, filename)
                data = pd.read_csv(file_path)
                all_data.append(data)
        if len(all_data) == 0:
            raise ValueError(f"The directory {csv_file} contains no CSV files.")
        df = pd.concat(all_data, ignore_index=True)
    else:
        if csv_file.endswith(".csv"):
            df = pd.read_csv(csv_file)
        else:
            raise ValueError("Provided file is not a valid CSV file.")
    return df


df1 = read_csv_file(sys.argv[1])
df1.to_csv(os.path.abspath(sys.argv[2]), index=False)

