import os,sys
import re
import ROOT
import uproot
import numpy as np
import pandas as pd
import random
from ROOT import TMath
import math
from ROOT import RooRealVar, RooGenericPdf, RooArgSet, RooArgList, RooDataHist, RooFit, TCanvas, TH1F, TRandom3, RooFormulaVar, RooGaussian

if len(sys.argv) < 7:
   print("Usage: python find_gaussian_peaks.py <input_file> <tree_name> <variable_name> <num_bins> <minRange> <maxRange> <output_path>")
else:
   input_file = sys.argv[1]
   tree_name = sys.argv[2]
   variable_name = sys.argv[3]
   num_bins = int(sys.argv[4])
   x_min = float(sys.argv[5])
   x_max = float(sys.argv[6])
# Open the file
pattern = r'run(\d+)_ov(\d+)_(\w+)_(\w+)_ch(\d+)'
tree_match = re.match(pattern, tree_name)
if tree_match:
    run = tree_match.group(1)    # "64"
    ov = int(tree_match.group(2))     # "2"
    run_type = tree_match.group(3)   # "main"
    sipm_type = tree_match.group(4)   # "tile"
    channel = tree_match.group(5) # "2"

pattern_variable = r'(\w+)_ch(\d+)'
variable_match = re.match(pattern_variable, variable_name)

if variable_match:
    variable_name_short = variable_match.group(1)
    tile = variable_match.group(2)

print(num_bins, x_min, x_max)
file1 = ROOT.TFile(input_file)
tree = file1.Get(tree_name)
hist = ROOT.TH1F("hist","hist", int(num_bins), float(x_min), float(x_max))
tree.Draw("{}>>hist".format(variable_name))
# Print the final result
spectrum = ROOT.TSpectrum()
n_peaks = spectrum.Search(hist, 0.2 , "", 0.05)
peaks_tspectrum = []
for i in range(n_peaks):
    peaks_tspectrum.append(float(spectrum.GetPositionX()[i]))
peaks_tspectrum.sort()
if len(peaks_tspectrum) >= 2:
    distance = peaks_tspectrum[1] - peaks_tspectrum[0]
else:
    distance = 10
#distance = 43.5
sigQ = RooRealVar("sigQ", "sigQ", x_min, x_max)
#data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), ROOT.RooFit.Import(hist))
data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), hist)
# Define the parameters of the distribution
lambda_ = RooRealVar("lambda", "lambda", 0.2, 0.05, 0.8)
mu = RooRealVar("mu", "mu", 2, 0.1, 5)

# Create RooFit variables for the observables and parameters
sigma0 = RooRealVar("sigma0", "sigma0", 5,1,10)
sigmak = RooRealVar("sigmak", "sigmak", 5,1,10)
#sigma1 = RooFormulaVar("sigma1", "TMath::Sqrt( pow(sigma0, 2) + 1 * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
#sigma2 = RooFormulaVar("sigma2", "TMath::Sqrt( pow(sigma0, 2) + 2 * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
#sigma3 = RooFormulaVar("sigma3", "TMath::Sqrt( pow(sigma0, 2) + 3 * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
#sigma4 = RooFormulaVar("sigma4", "TMath::Sqrt( pow(sigma0, 2) + 4 * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
#sigma5 = RooFormulaVar("sigma5", "TMath::Sqrt( pow(sigma0, 2) + 5 * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
#sigma6 = RooFormulaVar("sigma6", "TMath::Sqrt( pow(sigma0, 2) + 6 * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
#sigma7 = RooFormulaVar("sigma7", "TMath::Sqrt( pow(sigma0, 2) + 7 * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
sigma1 = RooRealVar("sigma1", "sigma1", 5,1,10)
sigma2 = RooRealVar("sigma2", "sigma2", 5,1,10)
sigma3 = RooRealVar("sigma3", "sigma3", 5,1,10)
sigma4 = RooRealVar("sigma4", "sigma4", 5,1,10)
sigma5 = RooRealVar("sigma5", "sigma5", 5,1,10)
sigma6 = RooRealVar("sigma6", "sigma6", 5,1,10)
sigma7 = RooRealVar("sigma7", "sigma7", 5,1,10)
sigma8 = RooRealVar("sigma8", "sigma8", 5,1,10)
sigma9 = RooRealVar("sigma9", "sigma9", 5,1,10)
sigma10 = RooRealVar("sigma10", "sigma10", 5,1,10)
# Define the Gaussian PDF
ped = RooRealVar("ped", "ped", 0, -2, 2)
gain = RooRealVar("gain", "gain", distance, distance *0.8, distance *1.2)
if ov > 4:
    n_param = 4 + 10
    formula2 = "exp(-mu) * exp(-pow(sigQ - ped, 2)/(2 * pow(sigma0, 2))) / sigma0  + \
                        mu * exp(-mu - 1 * lambda) * exp(-pow(sigQ - (ped + 1 * gain), 2)/(2 * pow(sigma1, 2))) / sigma1  + \
                        mu * pow(mu + 2 * lambda, 2-1) / TMath::Factorial(2) * exp(-mu - 2 * lambda) * exp(-pow(sigQ - (ped + 2 * gain), 2)/(2 * pow(sigma2, 2))) /  sigma2   + \
                        mu * pow(mu + 3 * lambda, 3-1) / TMath::Factorial(3) * exp(-mu - 3 * lambda) * exp(-pow(sigQ - (ped + 3 * gain), 2)/(2 * pow(sigma3, 2))) /  sigma3  + \
                        mu * pow(mu + 4 * lambda, 4-1) / TMath::Factorial(4) * exp(-mu - 4 * lambda) * exp(-pow(sigQ - (ped + 4 * gain), 2)/(2 * pow(sigma4, 2))) / sigma4  + \
                        mu * pow(mu + 5 * lambda, 5-1) / TMath::Factorial(5) * exp(-mu - 5 * lambda) * exp(-pow(sigQ - (ped + 5 * gain), 2)/(2 * pow(sigma5, 2))) / sigma5 + \
                        mu * pow(mu + 6 * lambda, 6-1) / TMath::Factorial(6) * exp(-mu - 6 * lambda) * exp(-pow(sigQ - (ped + 6 * gain), 2)/(2 * pow(sigma6, 2))) / sigma6 + \
                        mu * pow(mu + 7 * lambda, 7-1) / TMath::Factorial(7) * exp(-mu - 7 * lambda) * exp(-pow(sigQ - (ped + 7 * gain), 2)/(2 * pow(sigma7, 2))) / sigma7 + \
                        mu * pow(mu + 8 * lambda, 8-1) / TMath::Factorial(8) * exp(-mu - 8 * lambda) * exp(-pow(sigQ - (ped + 8 * gain), 2)/(2 * pow(sigma8, 2))) / sigma8 + \
                        mu * pow(mu + 9 * lambda, 9-1) / TMath::Factorial(9) * exp(-mu - 9 * lambda) * exp(-pow(sigQ - (ped + 9 * gain), 2)/(2 * pow(sigma9, 2))) / sigma9 + \
                        mu * pow(mu + 10 * lambda, 10-1) / TMath::Factorial(10) * exp(-mu - 10 * lambda) * exp(-pow(sigQ - (ped + 10 * gain), 2)/(2 * pow(sigma10, 2))) / sigma10"
    
    poisson_gen = RooGenericPdf("poisson_gen", formula2, RooArgList(lambda_, mu, sigQ, ped, gain, sigma0, sigma1, sigma2,sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9, sigma10))
else:
    n_param = 4 + 8
    formula2 = "exp(-mu) * exp(-pow(sigQ - ped, 2)/(2 * pow(sigma0, 2))) / sigma0  + \
                        mu * exp(-mu - 1 * lambda) * exp(-pow(sigQ - (ped + 1 * gain), 2)/(2 * pow(sigma1, 2))) / sigma1  + \
                        mu * pow(mu + 2 * lambda, 2-1) / TMath::Factorial(2) * exp(-mu - 2 * lambda) * exp(-pow(sigQ - (ped + 2 * gain), 2)/(2 * pow(sigma2, 2))) /  sigma2   + \
                        mu * pow(mu + 3 * lambda, 3-1) / TMath::Factorial(3) * exp(-mu - 3 * lambda) * exp(-pow(sigQ - (ped + 3 * gain), 2)/(2 * pow(sigma3, 2))) /  sigma3  + \
                        mu * pow(mu + 4 * lambda, 4-1) / TMath::Factorial(4) * exp(-mu - 4 * lambda) * exp(-pow(sigQ - (ped + 4 * gain), 2)/(2 * pow(sigma4, 2))) / sigma4  + \
                        mu * pow(mu + 5 * lambda, 5-1) / TMath::Factorial(5) * exp(-mu - 5 * lambda) * exp(-pow(sigQ - (ped + 5 * gain), 2)/(2 * pow(sigma5, 2))) / sigma5 + \
                        mu * pow(mu + 6 * lambda, 6-1) / TMath::Factorial(6) * exp(-mu - 6 * lambda) * exp(-pow(sigQ - (ped + 6 * gain), 2)/(2 * pow(sigma6, 2))) / sigma6 + \
                        mu * pow(mu + 7 * lambda, 7-1) / TMath::Factorial(7) * exp(-mu - 7 * lambda) * exp(-pow(sigQ - (ped + 7 * gain), 2)/(2 * pow(sigma7, 2))) / sigma7 + \
                        mu * pow(mu + 8 * lambda, 8-1) / TMath::Factorial(8) * exp(-mu - 8 * lambda) * exp(-pow(sigQ - (ped + 8 * gain), 2)/(2 * pow(sigma8, 2))) / sigma8"
    poisson_gen = RooGenericPdf("poisson_gen", formula2, RooArgList(lambda_, mu, sigQ, ped, gain, sigma0, sigma1, sigma2,sigma3, sigma4, sigma5, sigma6, sigma7, sigma8))
#sigQ.setRange("NormalizationRange", -20, 200)

#poisson_gen.setNormRangeOverride("NormalizationRange")
#data = poisson_gen.generate(RooArgSet(sigQ), 25000)

# Specify the optimization algorithm
#ROOT.RooFit.SetOptions(ROOT.RooFit.Minimizer("Minuit2"))  # Choose Minuit2 as the optimizer

# Fit the data
#result = poisson_gen.fitTo(data, ROOT.RooFit.Save())
# Specify the optimization algorithm
minimizer = ROOT.RooMinimizer(poisson_gen.createNLL(data))
minimizer.setMinimizerType("Minuit2")  # Choose Minuit2 as the optimizer

# Perform the fit
result = minimizer.fit("")
#final_params = result.floatParsFinal()
## Access the parameter values from the fit result
#floating_params = result.floatParsFinal()
# For a given pdf and variable x
#integral_pdf = poisson_gen.createIntegral(RooArgSet(sigQ))
#expected_event = poisson_gen.expectedEvents(RooArgSet(sigQ))

#print(integral_pdf.getVal(),data.sumEntries(), expected_event)
## Covariance matrix
#cov_matrix = result.covarianceMatrix()
#
## You can print the matrix as follows:
#cov_matrix.Print()
## Correlation matrix
#cor_matrix = result.correlationMatrix()
#
## Print the matrix
#cor_matrix.Print()
#final_params = result.floatParsFinal()
#
#for i in range(final_params.getSize()):
#    print("Index: ", i, "Parameter: ", final_params[i].GetName())
## Create a chi-squared variable from the pdf and the data
chi2 = ROOT.RooChi2Var("chi2", "chi2", poisson_gen, data)
# Get the chi-squared value
chi2_val = chi2.getVal()
print(chi2_val)
# Get number of degrees of freedom
ndf = data.numEntries() - n_param #result.floatParsFinal().getSize()
# Calculate chi-squared per degree of freedom
chi2_ndf = chi2_val / ndf
print(chi2_ndf)
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
frame.SetTitle(f"Charge spectrum fit of overvoltage {ov}V, Run {run}, Ch {channel}, Tile {tile}")
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

canvas.SaveAs("last.pdf")
print(distance, gain.getVal())
print(mu.getVal(), lambda_.getVal())
