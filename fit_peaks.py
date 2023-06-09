import ROOT
import os, sys
import re
import numpy as np
import scipy.stats as stats
from scipy.signal import argrelextrema
from collections import defaultdict
import pandas as pd

import json

ROOT.gSystem.Load("libRooFit")

var_dict = {"sigQ": "ADC", "sigAmp":"Amplitude", "dcrQ":"ADC"}
seq_list = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]

def get_gaussian_scale(mean, sigma, lower_bound, upper_bound):
    # Calculate the probability using the CDF
    probability = stats.norm.cdf(upper_bound, mean, sigma) - stats.norm.cdf(lower_bound, mean, sigma)
    return probability

def fit_single_gaussian_peak(input_file, tree_name, variable_name, peak_mean, peak_sigma, peak_seq):
    pattern = r'run(\d+)_ov(\d+)_(\w+)_(\w+)_ch(\d+)'

    tree_match = re.match(pattern, tree_name)
    if tree_match:
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
    file1 = ROOT.TFile(input_file)
    tree = file1.Get(tree_name)
    
    # Create RooFit variables
    x = ROOT.RooRealVar(variable_name, variable_name, peak_mean -  0.5 *peak_sigma, peak_mean + 0.5 * peak_sigma)
    data = ROOT.RooDataSet("data", "data", ROOT.RooArgSet(x), ROOT.RooFit.Import(tree))

    # Create the WorkSpace
    theWorkSpace = ROOT.RooWorkspace("theWorkSpace")
    # Create Gaussian model
    mean = ROOT.RooRealVar("mean", "mean", peak_mean, peak_mean - 1, peak_mean + 1)
    if sipm_type == "tile":
        sigma = ROOT.RooRealVar("sigma", "sigma", peak_sigma, peak_sigma * 0.1, peak_sigma * 1.2)
    elif sipm_type == "reff":
        sigma = ROOT.RooRealVar("sigma", "sigma", peak_sigma * 0.1, peak_sigma * 0.01, peak_sigma * 1.2)
    theWorkSpace.Import(x)
    theWorkSpace.Import(mean)
    theWorkSpace.Import(sigma)
    theWorkSpace.factory("RooGaussian::signal("+variable_name+", mean,sigma)")
    #theWorkSpace.factory("RooChebychev::background(" + variable_name + ", {a0[0.25,0,1], a1[-0.25,-1,1],a2[0.,-0.5,0.5]})")
    theWorkSpace.factory("RooExponential::background(" +variable_name+ ", lp[0,-10,10])")
    #theWorkSpace.factory("RooPolynomial::background(" + variable_name + ", {a0[0,-1,1],a1[0,-1,1], a2[0, -0.5,0.5] })")
    theWorkSpace.factory("SUM::final( amplitude[0.9, 0.5, 1] * signal, background)")
    # Fit the model
    result = theWorkSpace.pdf("final").fitTo(data, ROOT.RooFit.Save())

    # Create a TPaveText to display parameter values and uncertainties
    param_box = ROOT.TPaveText(0.6, 0.6, 0.9, 0.9, "NDC")
    param_box.SetFillColor(ROOT.kWhite)
    param_box.SetBorderSize(1)
    param_box.SetTextFont(42)
    param_box.SetTextSize(0.035)
    
    # Get post-fit values and uncertainties
    final_params = result.floatParsFinal()
    
    mean_postfit = final_params.find("mean")
    mean_value = mean_postfit.getVal()
    mean_error = mean_postfit.getError()
    
    sigma_postfit = final_params.find("sigma")
    sigma_value = sigma_postfit.getVal()
    sigma_error = sigma_postfit.getError()

    f_val = final_params.find("amplitude").getVal()
    f_error = final_params.find("amplitude").getError()

    total_entries = data.numEntries()
    # Create a normalization object for the Gaussian PDF
    gauss_norm = theWorkSpace.pdf("signal").createIntegral(ROOT.RooArgSet(x), ROOT.RooFit.NormSet(x), ROOT.RooFit.Range("x"))
    # Create a normalization object for the exponential PDF
    expo_norm = theWorkSpace.pdf("background").createIntegral(ROOT.RooArgSet(x), ROOT.RooFit.NormSet(x), ROOT.RooFit.Range("x"))

    # Calculate the total number of events predicted by the signal PDF
    total_events_signal = gauss_norm.getVal() * f_val * data.numEntries()
    total_events_background = expo_norm.getVal() * (1 - f_val) * data.numEntries()
    # Calculate the uncertainties of the signal and background number of events
    total_events_signal_unc = np.sqrt((total_entries * f_error)**2 + (f_val * total_entries)**2 * (f_error / f_val)**2)
    total_events_background_unc = np.sqrt((total_entries * f_error)**2 + ((1 - f_val) * total_entries)**2 * (f_error / (1 - f_val))**2)
    param_box.AddText(f"Initial Events: {total_entries:.0f}")
    param_box.AddText(f"Signal Events: {total_events_signal:.0f} #pm {total_events_signal_unc:.0f}")
    param_box.AddText(f"Bkg. Events: {total_events_background:.0f} #pm {total_events_background_unc:.0f}")
    param_box.AddText(f"Mean = {mean_value:.3f} #pm {mean_error:.3f}")
    param_box.AddText(f"Sigma = {sigma_value:.3f} #pm {sigma_error:.3f}")

    canvas = ROOT.TCanvas("c1","c1", 1200, 800)
    frame = x.frame()
    frame.SetXTitle(var_dict[variable_name_short])
    frame.SetYTitle("Events")
    frame.SetTitle(f"{var_dict[variable_name_short]} fit result of peak {peak_mean:.2f} (the {seq_list[peak_seq]} peak)")
    data.plotOn(frame)
    theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
    theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.Components("background"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
    frame.Draw()
    param_box.Draw("same")
    # Create TLine objects for the legend
    signal_line = ROOT.TLine()
    signal_line.SetLineWidth(4)
    signal_line.SetLineColor(ROOT.kBlue)
    signal_line.SetLineStyle(ROOT.kSolid)
    
    background_line = ROOT.TLine()
    background_line.SetLineWidth(4)
    background_line.SetLineColor(ROOT.kRed)
    background_line.SetLineStyle(ROOT.kDashed)
    
    # Create a TLegend and add entries for signal and background PDFs
    legend = ROOT.TLegend(0.1, 0.7, 0.3, 0.9)
    legend.SetTextSize(0.05)
    legend.AddEntry(signal_line, "S+B", "l")
    legend.AddEntry(background_line, "B only", "l")
    legend.Draw("same")
    canvas.SaveAs(f'{variable_name_short}_{run_type}_run{run}_ov{ov}_{sipm_type}_ch{channel}_po{tile}_peak{peak_seq}.pdf')

    print("Results:")
    print(f"  Peak: Mean = {mean_value}, Sigma = {sigma_value},  Events = {total_entries}")
    #fit_info = {}
    #fit_info['init_events'] = int(total_entries),
    #fit_info['signal_events'] = total_events_signal,
    #fit_info['signal_events_unc'] = total_events_signal_unc,
    #fit_info['bkg_events'] = total_events_background,
    #fit_info['bkg_events_unc'] = total_events_background_unc,
    #fit_info['sigma'] = float(sigma_value),
    #fit_info['sigma_error'] = sigma_error,
    #fit_info['mean'] = mean_value,
    #fit_info['mean_error'] = mean_error,
    #fit_info = []
    scale = get_gaussian_scale (mean_value, sigma_value, peak_mean -  0.5 *peak_sigma, peak_mean + 0.5 * peak_sigma)
    fit_info = {
                    'mean' : float(mean_value),
                    'mean_error' : float(mean_error),
                    'sigma' : float(sigma_value),
                    'sigma_error' : float(sigma_error),
                    'init_events' : int(total_entries),
                    'signal_events' : int(total_events_signal),
                    'signal_events_unc' : int(total_events_signal_unc),
                    'bkg_events' : int(total_events_background),
                    'gaussian_scale' : float(scale),
                    }
    
    return fit_info
# Function to calculate the autocorrelation of a signal
def calculate_autocorrelation(signal):
    autocorrelation = np.correlate(signal, signal, mode='full')
    return autocorrelation[len(autocorrelation)//2:]

# Function to find the repetition period (peak distance) from autocorrelation
def find_repetition_period(autocorrelation):
    repetition_period = np.argmax(autocorrelation[1:]) + 1
    return repetition_period

# Function to find the first and second peak distances from autocorrelation
def find_peak_distances(autocorrelation):
    # Find the first peak distance
    first_peak = np.argmax(autocorrelation[1:]) + 1

    # Remove the first peak by setting its value to 0
    autocorrelation[first_peak] = 0

    # Find the second peak distance
    second_peak = np.argmax(autocorrelation[1:]) + 1

    return first_peak, second_peak
def find_local_maximum(array, totalrange, window_size):
    n = len(array)
    
    # Apply smoothing to the array
    smoothed_array = np.convolve(array, np.ones(window_size)/window_size, mode='same')
    
    # Calculate the first derivative of the smoothed data
    first_derivative = np.gradient(smoothed_array)
    #first_derivative = np.gradient(array)
    
    # Calculate the second derivative of the smoothed data
    second_derivative = np.gradient(first_derivative)
    
    peaks = []
    for i in range(1, n - 1):
        if (
            array[i] > array[i - 1] and
            array[i] > array[i + 1] and
            first_derivative[i] > 0 and
            second_derivative[i] < 0
        ):
            peaks.append( i / n * totalrange )
    
    return peaks  # No local maximum found

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
    pattern = r'run(\d+)_ov(\d+)_(\w+)_(\w+)_ch(\d+)'
    print(tree_name)

    tree_match = re.match(pattern, tree_name)
    if tree_match:
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
    if len(sys.argv) == 8:
        output_path = sys.argv[7]
    else:
        output_path = f"results/{run_type}_run{run}/postfit"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # Create a histogram and fill it with the values from the TTree
    hist = ROOT.TH1F("hist", "Histogram of {}".format(variable_name), num_bins, minRange, maxRange)
    tree.Draw("{}>>hist".format(variable_name))
    n_entry = tree.GetEntries()
    # Print the final result
    spectrum = ROOT.TSpectrum()
    #########################  TSpectrum search ######################
    # (TH1, sigma , "options", threshold)
    n_peaks = spectrum.Search(hist, 0.5 , "", 0.1)
    if sipm_type == "reff":
        n_peaks = spectrum.Search(hist, 0.5 , "", 0.01)
    #if variable_name_short == "sigAmp" and int(ov) >= 4:
    if variable_name_short == "sigAmp":
        n_peaks = spectrum.Search(hist, 0.5 , "", 0.05)
    if variable_name_short == "sigQ" and int(ov) >= 5:
        n_peaks = spectrum.Search(hist, 0.5 , "", 0.05)
    ##################################################################
    peaks_tspectrum = []
    for i in range(n_peaks):
        peaks_tspectrum.append(float(spectrum.GetPositionX()[i]))
    peaks_tspectrum.sort()
    print("TSpectrum method:\n")
    for i, peak in enumerate(peaks_tspectrum):
        print("Peak {}: x = {}".format(i, peaks_tspectrum[i]))

    peaks_distance = peaks_tspectrum[1] - peaks_tspectrum[0]
    mean_tmp = 0
    combined_dict = defaultdict(list)
    for i, mean in enumerate(peaks_tspectrum):
        if i != 0:
            if abs(mean - mean_tmp) < 0.5 * peaks_distance:
                continue
        fit_info = fit_single_gaussian_peak(input_file, tree_name, variable_name, mean , peaks_distance, i)
        print(fit_info)
        fit_info['peak'] = i
        fit_info['events'] = n_entry
        fit_info['run_number'] = run
        fit_info['voltage'] = ov
        fit_info['channel'] = channel
        fit_info['type'] = sipm_type
        fit_info['position'] = tile 
        fit_info['run_type'] = run_type
        fit_info['var'] = variable_name_short
        mean_tmp = mean
        for key, value in fit_info.items():
            combined_dict[key].append(value)
    df = pd.DataFrame(combined_dict)
    # Save the DataFrame to a CSV file
    df.to_csv(f'{variable_name_short}_{run_type}_run{run}_ov{ov}_{sipm_type}_ch{channel}_po{tile}.csv', index=False)
    os.system(f"mv {variable_name_short}_{run_type}_run{run}_ov{ov}*.csv {output_path}")
    os.system(f"mv {variable_name_short}_{run_type}_run{run}_ov{ov}*.pdf {output_path}")
