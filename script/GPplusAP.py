import os,sys
import ROOT
import uproot
import numpy as np
from astropy.stats import knuth_bin_width
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
print(num_bins, x_min, x_max)
#file = uproot.open(input_file)

# Access the TTree
#tree = file[tree_name]

# Convert the branch to a Python list
#branch_data = tree[variable_name].array().tolist()
#list_Q = np.array(branch_data)
#bw = knuth_bin_width(list_Q)
#x_min, x_max = -30 ,220#list_Q.min(), list_Q.max()
file1 = ROOT.TFile(input_file)
tree = file1.Get(tree_name)
hist = ROOT.TH1F("hist","hist", int(num_bins), float(x_min), float(x_max))
sigQ = RooRealVar("sigQ", "sigQ", x_min, x_max)
tree.Draw("{}>>hist".format(variable_name))
data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), ROOT.RooFit.Import(hist))
# Define the parameters of the distribution
lambda_ = RooRealVar("lambda", "lambda", 0.2, 0.01, 0.3)
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
# Define the Gaussian PDF
ped = RooRealVar("ped", "ped", 0, -5, 5)
gain = RooRealVar("gain", "gain", 10, 5, 35)
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

for i in range(30):
    # Fit the data
    result = poisson_gen.fitTo(data, ROOT.RooFit.Save())
    final_params = result.floatParsFinal()
    # Access the parameter values from the fit result
    floating_params = result.floatParsFinal()
    for param in floating_params:
        # Get the parameter value, maximum, and minimum boundaries
        value = param.getVal()
        max_limit = param.getMax()
        min_limit = param.getMin()
    
        # Check if the parameter is at its maximum or minimum boundary
        if value >= max_limit:
            print(f"Parameter {param.GetName()} has reached its maximum boundary.")
        if value <= min_limit:
            print(f"Parameter {param.GetName()} has reached its minimum boundary.")
    
    print(mu.getVal(), lambda_.getVal())
    # For a given pdf and variable x
    integral_pdf = poisson_gen.createIntegral(RooArgSet(sigQ))
    expected_event = poisson_gen.expectedEvents(RooArgSet(sigQ))
    
    print(integral_pdf.getVal(),data.sumEntries(), expected_event)
    # Create a chi-squared variable from the pdf and the data
    chi2 = ROOT.RooChi2Var("chi2", "chi2", poisson_gen, data)
    # Get the chi-squared value
    chi2_val = chi2.getVal()
    print(chi2_val)
    # Get number of degrees of freedom
    ndf = data.numEntries() - result.floatParsFinal().getSize()
    # Calculate chi-squared per degree of freedom
    chi2_ndf = chi2_val / ndf
    print(chi2_ndf)
    if chi2_ndf < 2:
        break
    else:
        new_gain_val = random.randint(10,35)
        print("assign the new gain value:", new_gain_val)
        gain.setVal(new_gain_val)
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
# Plot the data and the fit
c = TCanvas("c", "c", 800, 600)
frame = sigQ.frame()
data.plotOn(frame)
poisson_gen.plotOn(frame)
frame.Draw()
c.SaveAs("last.png")


