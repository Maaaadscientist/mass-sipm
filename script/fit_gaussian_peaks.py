import ROOT
import os,sys
import numpy as np

import yaml
import json

ROOT.gSystem.Load("libRooFit")

def fit_single_gaussian_peak(input_file, tree_name, variable_name, peak_mean, peak_sigma, peak_seq):
    file1 = ROOT.TFile(input_file)
    tree = file1.Get(tree_name)

    # Create RooFit variables
    x = ROOT.RooRealVar(variable_name, variable_name, peak_mean - 4 * peak_sigma, peak_mean + 4 * peak_sigma)
    data = ROOT.RooDataSet("data", "data", ROOT.RooArgSet(x), ROOT.RooFit.Import(tree))

    # Create the WorkSpace
    theWorkSpace = ROOT.RooWorkspace("theWorkSpace")
    # Create Gaussian model
    mean = ROOT.RooRealVar("mean", "mean", peak_mean, peak_mean - peak_sigma, peak_mean + peak_sigma)
    sigma = ROOT.RooRealVar("sigma", "sigma", peak_sigma, peak_sigma * 0.1, peak_sigma * 2)
    width = ROOT.RooRealVar("width", "width", peak_sigma, peak_sigma * 0.5, peak_sigma * 2)
    theWorkSpace.Import(x)
    theWorkSpace.Import(mean)
    theWorkSpace.Import(width)
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
    frame.SetXTitle("ADC")
    frame.SetYTitle("Events")
    frame.SetTitle("fit result of peak" + str(peak_mean))
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
    canvas.SaveAs(str(tree_name) + "_"+ str(peak_seq) + "_" + str(peak_mean) +".pdf")

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
    fit_info = []
    fit_info.append({
                    'init_events' : int(total_entries),
                    'signal_events' : int(total_events_signal),
                    'signal_events_unc' : int(total_events_signal_unc),
                    'bkg_events' : int(total_events_background),
                    'mean' : mean_value,
                    'mean_error' : mean_error,
                    'sigma' : float(sigma_value),
                    'sigma_error' : float(sigma_error),
                    })
    
    return fit_info

def find_gaussian_peaks(input_file, tree_name, variable_name, num_bins, minRange, maxRange):
    # Load the TTree from the input ROOT file
    file1 = ROOT.TFile(input_file)
    tree = file1.Get(tree_name)

    # Create a histogram and fill it with the values from the TTree
    hist = ROOT.TH1F("hist", "Histogram of {}".format(variable_name), num_bins, minRange, maxRange)
    tree.Draw("{}>>hist".format(variable_name))
    spectrum = ROOT.TSpectrum()
    n_peaks = spectrum.Search(hist)

    peak_info = []
    for i in range(2):
                peak_info.append({
                    'mean': float(spectrum.GetPositionX()[i]),
                    'sigma': float(spectrum.GetPositionX()[1] - spectrum.GetPositionX()[0]),
                })

    return peak_info

def extract_roofit_pdf(x_var, pdf, n_points=1000):
    x_vals = np.linspace(x_var.getMin(), x_var.getMax(), n_points)
    pdf_vals = np.empty(len(x_vals), dtype=float)
    for i, x in enumerate(x_vals):
        x_var.setVal(x)
        pdf_vals[i] = pdf.getVal(ROOT.RooArgSet(x_var))
    return x_vals, pdf_vals


def get_histogram(input_file, tree_name, variable_name, num_bins, minRange, maxRange):
    # Load the TTree from the input ROOT file
    file1 = ROOT.TFile(input_file)
    tree = file1.Get(tree_name)

    # Create a histogram and fill it with the values from the TTree
    hist = ROOT.TH1F("hist", "Histogram of {}".format(variable_name), num_bins, minRange, maxRange)
    tree.Draw("{}>>hist".format(variable_name))

    return ut.getBinCenters(hist), ut.getBinWidths(hist), ut.getBinContents(hist), ut.getBinErrors(hist)

def extract_element(filepath):
    # Open the JSON file in read mode
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Loop through each item in the dictionary
    for key, value in data.items():
        # Loop through each key in the nested dictionary
        for subkey, subvalue in value.items():
            # Check if the value is a list with a single item
            if isinstance(subvalue, list) and len(subvalue) == 1:
                # Replace the list with its single item
                data[key][subkey] = subvalue[0]
    
    # Write the updated data back to the JSON file
    with open(filepath, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python find_gaussian_peaks.py <input_file> <tree_name> <variable_name> <num_bins> <minRange> <maxRange> <factor>")
    else:
        input_file = sys.argv[1]
        tree_name = sys.argv[2]
        variable_name = sys.argv[3]
        num_bins = int(sys.argv[4])
        minRange = float(sys.argv[5])
        maxRange = float(sys.argv[6])
        peak_info = find_gaussian_peaks(input_file, tree_name, variable_name, num_bins, minRange, maxRange)
        name_without_voltage = "_".join(tree_name.split("_")[:-1])
        output_directory = 'outputs/' + name_without_voltage
        if not os.path.isdir("outputs"): # check if outputs directory doesn't exist
            os.mkdir("outputs")
        if not os.path.isdir(output_directory): # check if directory doesn't exist
            os.mkdir(output_directory) # create directory
        file_to_write = open(output_directory + "/postfit_" + tree_name  + ".json","w")
        json_dict = {}
        for i, peak in enumerate(peak_info):
            print(f"Peak {i + 1}:")
            print(f"  Mean value: {peak['mean']}")
            print(f"  sigma: {peak['sigma']}")
            fit_info = fit_single_gaussian_peak(input_file, tree_name, variable_name, peak['mean'], peak['sigma'], i + 1)
            json_dict[i+1] = fit_info
        json_object = json.dumps(json_dict, indent=4)
        file_to_write.write(json_object)
        os.system("mv *.pdf " + output_directory)
        # Open the YAML file in write mode
        with open(output_directory + "/postfit_" + tree_name  + ".yaml", 'w') as yfile:
        # Dump the dictionary into the YAML file
            yaml.dump(json_dict, yfile)        


