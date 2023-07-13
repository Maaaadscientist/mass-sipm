import pandas as pd
import math
import re
import os, sys
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from copy import deepcopy
#from PyPDF2 import PdfMerger

import yaml
import ROOT

import statsmodels.api as sm


if len(sys.argv) < 5:
    print("Usage: python prepare_jobs <mother_dir> <run_info> <output_dir>")
else:
   input_tmp = sys.argv[1]
   yaml_path = sys.argv[2]
   run_info = sys.argv[3]
   output_tmp = sys.argv[4]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
input_dir =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
output_dir = os.path.abspath(output_tmp)
# Find a certain element based on other column values

pattern = "(\w+)_run_(\d+)"
matched = re.match(pattern, run_info)
if matched:
    run_type = matched.group(1)
    run_number = int(matched.group(2))
if not os.path.isdir(output_dir + "/csv"):
    os.makedirs(output_dir + "/csv")
if not os.path.isdir(output_dir + "/pdf"):
    os.makedirs(output_dir + "/pdf")


# Specify the path to your YAML file
yaml_path = os.path.abspath(yaml_path)

with open(yaml_path, "r") as afile:
    yaml_data = yaml.safe_load(afile)
# Access and manipulate the YAML data
if isinstance(yaml_data, dict):
    light_run_number = yaml_data.get(run_number)
else:
    print("Invalid YAML data")

vbd_dir = input_dir + "/vbd/" + run_info + "/csv"
signal_dir = input_dir + "/signal-fit/" + run_info + "/csv"
light_dir = input_dir +"/light-fit/" + f"light_run_{light_run_number}" + "/csv"
dcr_dir = input_dir + "/dcr/" + run_info + "/csv"
# Read the CSV file into a pandas DataFrame
if not os.path.isdir(vbd_dir):
    df = pd.read_csv(vbd_dir)
else:
    all_data = []
    for filename in os.listdir(vbd_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(vbd_dir, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)
    df = pd.concat(all_data, ignore_index=True)

for po in range(16):
    for ch in range(1, 17):
        
        filtered_df = df.loc[ (df['channel'] == ch) &
                        (df['position'] == po) ]
        vbd = filtered_df.head(1)['vbd'].values[0]
        vbd_err = filtered_df.head(1)['vbd_err'].values[0]
        #gain = filtered_df.head(1)['gain'].values[0]
        print(po,ch,vbd,vbd_err)

    
      
