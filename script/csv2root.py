import os, sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python prepare_jobs <input_dir>")
else:
   input_tmp = sys.argv[1]

csv_dir = os.path.abspath(input_tmp + "/csv")
root_dir = os.path.abspath(input_tmp + "/root")
name_short = input_tmp.split("/")[-1]
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

all_data = []

csv_count = 0
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_dir, filename)
        data = pd.read_csv(file_path)
        all_data.append(data)
        csv_count += 1

df = pd.concat(all_data, ignore_index=True)

h2 = ROOT.TH2F(name_short, name_short + " reference mu", 16, 0, 16, csv_count, 0, csv_count)
for po in range(16):
    for point in range(1, csv_count + 1):
        filtered_df = df.loc[(df['position'] == po) &
                             (df['point'] == point)]
        mu = filtered_df.head(1)['mu'].values[0]
        mu_err = filtered_df.head(1)['mu_err'].values[0]
        h2.SetBinContent(po + 1, point, mu)
        h2.SetBinError(po + 1, point, mu_err) 
f1 = ROOT.TFile(f"{root_dir}/{name_short}_reff_mu.root", "recreate")
h2.Write()
f1.Close()



