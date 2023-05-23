import ROOT
import os, sys
import numpy as np
from scipy.signal import argrelextrema

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
    if len(sys.argv) < 6:
        print("Usage: python find_gaussian_peaks.py <input_file> <tree_name> <variable_name> <num_bins> <minRange> <maxRange>")
    else:
        input_file = sys.argv[1]
        tree_name = sys.argv[2]
        variable_name = sys.argv[3]
        num_bins = int(sys.argv[4])
        minRange = float(sys.argv[5])
        maxRange = float(sys.argv[6])
    file1 = ROOT.TFile(input_file)
    tree = file1.Get(tree_name)
    # Create a histogram and fill it with the values from the TTree
    hist = ROOT.TH1F("hist", "Histogram of {}".format(variable_name), num_bins, minRange, maxRange)
    tree.Draw("{}>>hist".format(variable_name))
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
    h1 = ROOT.TH1F("test","test", num_bins, 0, maxRange - minRange)
    autocorrelation_array = []
    for i in range(hist.GetNbinsX() - 1):
        h1.SetBinContent(i, ifft_result[i*2] / num_bins / num_bins)
        autocorrelation_array.append(ifft_result[i*2]/num_bins )
    c1 = ROOT.TCanvas("c1","c1",600,600)
    spectrum = ROOT.TSpectrum()
    c1.Clear()
    spectrum_auto = ROOT.TSpectrum()
    n_peaks_auto = spectrum_auto.Search(h1, 2 , "", 0.2)
    c1.SaveAs("autocorr.pdf")
    c1.Clear()
    #########################  TSpectrum search ######################
    # (TH1, sigma , "options", threshold)
    n_peaks = spectrum.Search(hist, 0.5 , "", 0.2)
    ##################################################################
    c1.SaveAs("tspectrum.pdf")
    peaks_tspectrum = []
    peaks_autoplustspectrum = []
    for i in range(n_peaks):
        peaks_tspectrum.append(float(spectrum.GetPositionX()[i]))
    for i in range(n_peaks_auto):
        peaks_autoplustspectrum.append(float(spectrum_auto.GetPositionX()[i]))
    peaks_tspectrum.sort()
    peaks_autoplustspectrum.sort()
    peaks_autocorrelation = find_local_maximum(autocorrelation_array, maxRange - minRange, 2)
    peaks_autocorrelation.sort()
    print("autocorrelation method:\n")
    for i, peak in enumerate(peaks_autocorrelation):
        print("Peak {}: x = {}".format(i, peaks_autocorrelation[i]))
    print("TSpectrum method:\n")
    for i, peak in enumerate(peaks_tspectrum):
        print("Peak {}: x = {}".format(i, peaks_tspectrum[i]))
    print("Combined method:\n")
    for i, peak in enumerate(peaks_autoplustspectrum):
        print("Peak {}: x = {}".format(i, peaks_autoplustspectrum[i]))
    # Assuming 'array' is your 1D numpy array
    #indices = argrelextrema(np.array(autocorrelation_array), np.greater)

    # The indices of the local maxima
    #print(indices[0] / len(autocorrelation_array) * (maxRange - minRange))
