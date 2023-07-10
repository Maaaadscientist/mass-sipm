import os,sys
import ROOT
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from astropy.stats import knuth_bin_width
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

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
tree.Draw("{}>>hist".format(variable_name))
#data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), ROOT.RooFit.Import(hist))
# Define the parameters of the distribution
# For a given pdf and variable x
#integral_pdf = poisson_gen.createIntegral(RooArgSet(sigQ))
#expected_event = poisson_gen.expectedEvents(RooArgSet(sigQ))

#print(integral_pdf.getVal(),data.sumEntries(), expected_event)
## Create a chi-squared variable from the pdf and the data
#chi2 = ROOT.RooChi2Var("chi2", "chi2", poisson_gen, data)
## Get the chi-squared value
#chi2_val = chi2.getVal()
#print(chi2_val)
## Get number of degrees of freedom
#ndf = data.numEntries() - result.floatParsFinal().getSize()
## Calculate chi-squared per degree of freedom
#chi2_ndf = chi2_val / ndf
#print(chi2_ndf)
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


histo = []
for i in range(num_bins):
    histo.append(hist.GetBinContent(i+1))
# Replace this with your spectrum histogram data
arr = np.array(histo)

# Perform FFT
fft_result = np.fft.fft(arr)

# Multiply by complex conjugate
mult_result = fft_result * np.conjugate(fft_result)

# Perform IFFT on the multiplied results
ifft_result = np.fft.irfft(mult_result)

# Print the final result
autocorrelation_array = []
for i in range(hist.GetNbinsX() - 1):
    print(ifft_result[i*2])
    autocorrelation_array.append(ifft_result[i*2]/num_bins / num_bins )
x = np.array(range(hist.GetNbinsX() - 1))
y = np.array(autocorrelation_array)
spline = CubicSpline(x, y)
# Generate x values within the range of the data
x_vals = np.linspace(x.min(), x.max(), 100)

# Evaluate the spline function at the generated x values
y_vals = spline(x_vals)

# Plot the data points and the spline function
#plt.plot(x, y, 'o', label='Data Points')
#plt.plot(x_vals, y_vals, label='Spline Function')
#plt.legend()
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Spline Function')
#plt.grid(True)
#plt.savefig("comp.pdf")
#plt.close()
# Define the derivative of the spline function
spline_derivative = spline.derivative()

# Find the critical points (where derivative equals zero)
critical_points = spline_derivative.roots()

# Initialize lists to store local maximum points
local_maximums = []

# Iterate through the critical points and check if they are local maximums
for point in critical_points:
    if spline_derivative(point - 0.01) > 0 and spline_derivative(point + 0.01) < 0:
        local_maximums.append((point, spline(point)))

print(local_maximums)
# Sort the local maximum points based on their y-values in descending order
local_maximums.sort(key=lambda x: x[1], reverse=True)

# Print the first and second local maximum points
if len(local_maximums) >= 1:
    print("First Local Maximum at x =", local_maximums[0][0])
    print("Value at First Local Maximum y =", local_maximums[0][1])

if len(local_maximums) >= 2:
    print("Second Local Maximum at x =", local_maximums[1][0])
    print("Value at Second Local Maximum y =", local_maximums[1][1])
