import os,sys
import re
import ROOT
import uproot
import numpy as np
import pandas as pd
import random
from collections import defaultdict 
from ROOT import TMath
import math
from ROOT import RooRealVar, RooGenericPdf, RooArgSet, RooArgList, RooDataHist, RooFit, TCanvas, TH1F, TRandom3, RooFormulaVar, RooGaussian

ov_peaks = {1:5, 2:7, 3:8, 4:9, 5:10, 6: 12}
ov_ranges = {1:100, 2:200, 3:250, 4:320, 5:420, 6: 600}
def GPfunction(i):
    return f"mu * pow(mu + {i} * lambda, {i}-1) / TMath::Factorial({i}) * exp(-mu - {i} * lambda) * exp(-pow(sigQ - (ped + {i} * gain), 2)/(2 * pow(sigma{i}, 2))) /  sigma{i}"
if len(sys.argv) < 4:
   print("Usage: python find_gaussian_peaks.py <input_file> <tree_name> <variable_name> <output_path> ")
else:
   input_file = sys.argv[1]
   tree_name = sys.argv[2]
   variable_name = sys.argv[3]
# Open the file
pattern_variable = r'(\w+)_ch(\d+)'
variable_match = re.match(pattern_variable, variable_name)

filename = input_file.split("/")[-1]
print(filename)
if variable_match:
    variable_name_short = variable_match.group(1)
    tile = variable_match.group(2)
pattern_name = r'(\w+)_run_(\w+)_ov_(\d+).00_sipmgr_(\d+)_(\w+)'
name_match = re.match(pattern_name, filename)
if name_match:
    run = str(name_match.group(2))
    ov = int(name_match.group(3))
    channel = int(name_match.group(4))
    sipm_type = name_match.group(5)
    print(ov)
if len(sys.argv) == 5:
    output_path = sys.argv[4]
else:
    output_path = f"results/main_run_{run}"

if not os.path.isdir(output_path + "/plots"):
    os.makedirs(output_path + "/plots")
if not os.path.isdir(output_path + "/csv"):
    os.makedirs(output_path + "/csv")
#num_bins = 40 + 80 * ov
num_bins = ov_ranges[ov]
x_max = num_bins
file1 = ROOT.TFile(input_file)
tree = file1.Get(tree_name)
n_entry = tree.GetEntries()
tree.Draw("{}>>histogram".format(f"baselineQ_ch{tile}"))
histogram = ROOT.gPad.GetPrimitive("histogram")
baseline = histogram.GetMean()
baseline_res = histogram.GetRMS()
hist = ROOT.TH1F("hist","hist", int(num_bins), baseline - baseline_res * 5, baseline + float(x_max))
tree.Draw("{}>>hist".format(variable_name))
# Print the final result
spectrum = ROOT.TSpectrum()
n_peaks = spectrum.Search(hist, 0.2 , "", 0.05)
peaks_tspectrum = []
for i in range(n_peaks):
    peaks_tspectrum.append(float(spectrum.GetPositionX()[i]))
peaks_tspectrum.sort()
if len(peaks_tspectrum) >= 2 and peaks_tspectrum[1] - peaks_tspectrum[0] > 8:
    distance = peaks_tspectrum[1] - peaks_tspectrum[0]
else:
    distance = 10
#distance = 43.5
sigQ = RooRealVar("sigQ", "sigQ", baseline - baseline_res * 5, baseline + float(x_max))
#data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), ROOT.RooFit.Import(hist))
data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), hist)
# Define the parameters of the distribution
lambda_ = RooRealVar("lambda", "lambda", 0.2, 0.05, 0.8)
mu = RooRealVar("mu", "mu", 2, 0.1, 5)

for i in range(16):
    globals()[f'sigma{i}'] = RooRealVar(f"sigma{i}", "sigma{i}", 5,1,10)
# Define the Gaussian PDF
ped = RooRealVar("ped", "ped", baseline, baseline - baseline_res *2, baseline + baseline_res * 2)
gain = RooRealVar("gain", "gain", distance, distance *0.8, distance *1.2)

n_param = 4 + ov_peaks[ov]
formula2 = f"exp(-mu) * exp(-pow(sigQ - ped, 2)/(2 * pow(sigma0, 2))) / sigma0  + \
                    mu * exp(-mu - 1 * lambda) * exp(-pow(sigQ - (ped + 1 * gain), 2)/(2 * pow(sigma1, 2))) / sigma1 + "
argList = RooArgList(lambda_, mu, sigQ, ped, gain, sigma0, sigma1)
for i in range(2, 2 + ov_peaks[ov]):
    argList.add(globals()[f"sigma{i}"])
    formula2 += f"{GPfunction(i)}"
    if i != range(2, 2 + ov_peaks[ov])[-1]:
        formula2 += "+"
  
poisson_gen = RooGenericPdf("poisson_gen", formula2, argList)
# Specify the optimization algorithm
minimizer = ROOT.RooMinimizer(poisson_gen.createNLL(data))
minimizer.setMinimizerType("Minuit2")  # Choose Minuit2 as the optimizer

# Perform the fit
result = minimizer.fit("")
## Create a chi-squared variable from the pdf and the data
chi2 = ROOT.RooChi2Var("chi2", "chi2", poisson_gen, data)
# Get the chi-squared value
chi2_val = chi2.getVal()
# Get number of degrees of freedom
ndf = data.numEntries() - n_param #result.floatParsFinal().getSize()
# Calculate chi-squared per degree of freedom
chi2_ndf = chi2_val / ndf
# Plot the data and the fit
canvas = ROOT.TCanvas("c1","c1", 1200, 800)
# Divide the canvas into two asymmetric pads
pad1 =ROOT.TPad("pad1","This is pad1",0.05,0.05,0.72,0.97);
pad2 = ROOT.TPad("pad2","This is pad2",0.72,0.05,0.98,0.97);
pad1.Draw()
pad2.Draw()
pad1.cd()
frame = sigQ.frame()
frame.SetXTitle("Charge")
frame.SetYTitle("Events")
frame.SetTitle(f"Charge spectrum fit of overvoltage (Run {run} ov {ov}V ch{channel} tile{tile})")
data.plotOn(frame)
#theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
#theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.Components("background"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
poisson_gen.plotOn(frame)
frame.Draw()
# Create TLine objects for the legend
pad2.cd()
# Create a TPaveText to display parameter values and uncertainties
param_box = ROOT.TPaveText(0.01, 0.9, 0.9, 0.1, "NDC")
param_box.SetFillColor(ROOT.kWhite)
param_box.SetBorderSize(1)
param_box.SetTextFont(42)
param_box.SetTextSize(0.08)
param_box.AddText(f"#mu = {mu.getVal():.3f} #pm {mu.getError():.3f}")
param_box.AddText(f"#lambda = {lambda_.getVal():.3f} #pm {lambda_.getError():.3f}")
param_box.AddText(f"ped = {ped.getVal():.3f} #pm {ped.getError():.3f}")
param_box.AddText(f"gain = {gain.getVal():.3f} #pm {gain.getError():.3f}")
param_box.AddText(f"#sigma0 = {sigma0.getVal():.3f} #pm {sigma0.getError():.3f}")
param_box.AddText(f"#sigma1 = {sigma1.getVal():.3f} #pm {sigma1.getError():.3f}")
param_box.AddText(f"#sigma2 = {sigma2.getVal():.3f} #pm {sigma2.getError():.3f}")
param_box.AddText(f"#sigma3 = {sigma3.getVal():.3f} #pm {sigma3.getError():.3f}")
param_box.AddText(f"#sigma4 = {sigma4.getVal():.3f} #pm {sigma4.getError():.3f}")
param_box.AddText(f"#sigma5 = {sigma5.getVal():.3f} #pm {sigma5.getError():.3f}")
param_box.AddText(f"#chi2/NDF = {chi2_ndf:.3f}")
param_box.Draw("same")

canvas.SaveAs(f"{output_path}/plots/{filename.replace('.root', '')}_po{tile}.pdf")
fit_info = {
                'mu' : float(mu.getVal()),
                'mu_err' : float(mu.getError()),
                'lambda' : float(lambda_.getVal()),
                'lambda_err' : float(lambda_.getError()),
                'ped' : float(ped.getVal()),
                'ped_err' : float(ped.getError()),
                'gain' : float(gain.getVal()),
                'gain_err' : float(gain.getError()),
                'chi2' : chi2_ndf,
                'sigma0' : float(sigma0.getVal()),
                'sigma1' : float(sigma1.getVal()),
                'sigma2' : float(sigma2.getVal()),
                'sigma3' : float(sigma3.getVal()),
                'sigma4' : float(sigma4.getVal()),
                'sigma5' : float(sigma5.getVal()),
                'sigma6' : float(sigma6.getVal()),
                'sigma7' : float(sigma7.getVal()),
                'sigma8' : float(sigma8.getVal()),
                'sigma9' : float(sigma9.getVal()),
                'sigma0_err' : float(sigma0.getError()),
                'sigma1_err' : float(sigma1.getError()),
                'sigma2_err' : float(sigma2.getError()),
                'sigma3_err' : float(sigma3.getError()),
                'sigma4_err' : float(sigma4.getError()),
                'sigma5_err' : float(sigma5.getError()),
                'sigma6_err' : float(sigma6.getError()),
                'sigma7_err' : float(sigma7.getError()),
                'sigma8_err' : float(sigma8.getError()),
                'sigma9_err' : float(sigma9.getError()),
                }
fit_info['events'] = n_entry
fit_info['run_number'] = run
fit_info['voltage'] = ov
fit_info['channel'] = channel
fit_info['position'] = tile
fit_info['baseline_res'] = baseline_res
print(distance, gain.getVal())
print(mu.getVal(), lambda_.getVal())
print(len(peaks_tspectrum))
combined_dict = defaultdict(list)
for key, value in fit_info.items():
    combined_dict[key].append(value)
df = pd.DataFrame(combined_dict)
# Save the DataFrame to a CSV file
df.to_csv(f"{output_path}/csv/{filename.replace('.root', '')}_po{tile}.csv", index=False)
