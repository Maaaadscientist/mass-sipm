import os, sys
import re
import math
import ROOT
import pandas as pd

if len(sys.argv) < 5:
    raise OSError("Usage: python main_run_light_fit.py <inputFile> <treeName> <branchName> <outputPath>")
else:
    input_tmp = sys.argv[1]
    tree_name = sys.argv[2]
    branch_name = sys.argv[3]
    output_tmp = sys.argv[4]

input_path = os.path.abspath(input_tmp)
output_path = os.path.abspath(output_tmp)
if not os.path.isdir(f"{output_path}/pdf"):
    os.makedirs(f"{output_path}/pdf")
if not os.path.isdir(f"{output_path}/csv"):
    os.makedirs(f"{output_path}/csv")

pattern_name = r'(\w+)_run_(\w+)_ov_(\d+).00_sipmgr_(\d+)_(\w+)'
name_match = re.match(pattern_name, filename)
if name_match:
    run = str(name_match.group(2))
    ov = int(name_match.group(3))
    channel = int(name_match.group(4))
    sipm_type = name_match.group(5)
if "/" in input_path:
    filename = input_path.split("/")[-1]
else:
    filename = input_path
f1 = ROOT.TFile(input_path)
tree = f1.Get(tree_name)
mu_list = []
mu_err_list = []
position_list = []
run_list = []
channel_list = []

for po in range(16):
    tree.Draw(f"baselineQ_ch{po}>>histogram{po}")
    histogram = ROOT.gPad.GetPrimitive(f"histogram{po}") 
    baseline = histogram.GetMean()
    baseline_sigma = histogram.GetRMS()
    hist = ROOT.TH1F(f"chargehist{po}", f"1st peak of light charge hist (run{run} channel{channel} ov{ov}V)", 500, baseline - 3 * baseline_sigma, baseline + 3 * baseline_sigma)
    canvas = ROOT.TCanvas("c1","c1", 1200, 800)
    tree.Draw(f"{branch_name}_ch{po}>>chargehist{po}")
    if po == 0:
        canvas.Print(f"{output_path}/pdf/charge_fit_reff_ch{channel}_ov{ov}.pdf[")
    else:
        canvas.Print(f"{output_path}/pdf/charge_fit_reff_ch{channel}_ov{ov}.pdf")
    #hist = ROOT.gPad.GetPrimitive("chargehist")
    
    # Assuming hist.Integral() and tree.GetEntries() are known values
    integral = hist.Integral()
    entries = tree.GetEntries()
    
    # Assuming the uncertainties on hist.Integral() and tree.GetEntries() are known
    delta_integral = math.sqrt(hist.Integral())  # Enter the uncertainty of hist.Integral()
    delta_entries = math.sqrt(tree.GetEntries())  # Enter the uncertainty of tree.GetEntries()
    
    # Calculate the partial derivatives
    d_mu_d_integral = -1 / integral
    d_mu_d_entries = 1 / entries
    
    # Calculate the error of mu using the error propagation formula
    delta_mu = math.sqrt((d_mu_d_integral * delta_integral)**2 + (d_mu_d_entries * delta_entries)**2)

    mu = -ROOT.TMath.Log(integral/entries)
    print(mu, delta_mu)
    mu_list.append(mu)
    mu_err_list.append(delta_mu)
    position_list.append(po)
    run_list.append(run)
    ov_list.append(ov)
    channel_list.append(channel)

df = pd.DataFrame()
df['mu'] = mu_list
df['mu_err'] = mu_err_list
df['position'] = position_list
df['run_number'] = run_list
df['channel'] = channel_list
df['voltage'] = ov_list
df.to_csv(f"{output_path}/csv/charge_fit_reff_ch{channel}_ov{ov}.csv", index=False)
canvas.Print(f"{output_path}/pdf/charge_fit_reff_ch{channel}_ov{ov}.pdf]")
