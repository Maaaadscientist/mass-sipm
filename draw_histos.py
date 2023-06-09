import ROOT
import os, sys
import re
import numpy as np



var_dict = {"sigQ": "ADC", "sigAmp":"Amplitude", "dcrQ":"ADC"}
physics_dict = {"sigQ": "Charge", "sigAmp":"Amplitude"}
seq_list = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python find_gaussian_peaks.py <input_file> <tree_name> <variable_name> <num_bins> <minRange> <maxRange> <output_path>")
    else:
        input_file = sys.argv[1]
        tree_name = sys.argv[2]
        variable_name = sys.argv[3]
        num_bins = int(sys.argv[4])
        minRange = float(sys.argv[5])
        maxRange = float(sys.argv[6])
    file1 = ROOT.TFile(input_file)
    tree = file1.Get(tree_name)
    pattern = r'run(\d+)_ov(\d+)_(\w+)_(\w+)_(\d+)'
    print(tree_name)

    tree_match = re.match(pattern, tree_name)
    if tree_match:
        print("successfully matched")
        run = tree_match.group(1)    # "64"
        ov = tree_match.group(2)     # "2"
        run_type = tree_match.group(3)   # "main"
        sipm_type = tree_match.group(4)   # "tile"
        channel = tree_match.group(5) # "2"

    pattern_variable = r'(\w+)_ch(\d+)'
    variable_match = re.match(pattern_variable, variable_name)

    if variable_match:
        variable_name_short = variable_match.group(1)
        tile = variable_match.group(2)
    sipm_po = int(tile) +1
    if len(sys.argv) == 8:
        output_path = sys.argv[7]
    else:
        output_path = f"results/{run_type}_run{run}/postfit"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # Create a histogram and fill it with the values from the TTree
    hist = ROOT.TH1F("hist", f"{physics_dict[variable_name_short]} spectrum  ({run_type} run-{run} ch-{channel} SiPM-{sipm_po} ov-{ov}V)", num_bins, minRange, maxRange)
    hist.SetStats(ROOT.kFALSE);
    hist.GetXaxis().SetTitle(var_dict[variable_name_short]);
    hist.GetYaxis().SetTitle("Events");
    c1 = ROOT.TCanvas("c1","c1",600,600)
    tree.Draw("{}>>hist".format(variable_name))
    c1.SaveAs(f"hist_{variable_name_short}_{run_type}_run{run}_ov{ov}_{sipm_type}_ch{channel}_po{tile}.pdf")
    os.system(f"mv hist_{variable_name_short}_{run_type}_run{run}_ov{ov}*.pdf {output_path}")
