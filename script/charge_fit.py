import os
import sys
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import ROOT
from ROOT import (TMath, RooRealVar, RooGenericPdf, RooArgSet, RooDataHist, 
                  RooFit, TCanvas, TH1F, TRandom3, RooFormulaVar, RooGaussian,
                  RooArgList)

ov_peaks = {1: 6, 2: 9, 3: 8, 4: 10, 5: 12, 6: 14}
ov_ranges = {1: 100, 2: 200, 3: 250, 4: 320, 5: 420, 6: 600}

def gauss(x, x_k, sigma_k):
    return f"(1/{sigma_k} * TMath::Exp(- TMath::Power({x} - {x_k}, 2)/(2 * TMath::Power({sigma_k}, 2))))"

def peak_pos(k, ped, gain):
    if k < 0 or type(k) is not int:
        raise ValueError("k should be a non-negative integer!")
    return f"({ped} + {k} * {gain})"

def generalized_poisson(k, mu, lambda_):
    exp_term = f"TMath::Exp(-({mu} + {k} * {lambda_}))"
    main_term = f"{mu} * TMath::Power(({mu} + {k} * {lambda_}), {k}-1)"
    factorial_term = math.factorial(k)
    if type(k) is not int:
        raise ValueError("k should be an integer!") 
    return f"({main_term} * {exp_term} / {factorial_term})"

def prepare_GP_pdf(n):
    if n < 1 or type(n) is not int:
        raise ValueError("k should be an positive interger")
    formula = ""
    GP0 = generalized_poisson(0, "mu", "lambda")
    x_0 = peak_pos(0, "ped", "gain")
    Gauss0 = gauss("sigQ", x_0, "sigma0")
    formula += f"{GP0} * {Gauss0}"
    for k in range(1, n+1):
        GPk = generalized_poisson(k, "mu", "lambda")
        x_k = peak_pos(k, "ped", "gain")
        Gaussk = gauss("sigQ", x_k, f"sigma{k}")
        formula += f"+ {GPk} * {Gaussk}"
    return formula

def calculate_GP(k, mu, lambda_):
    formula_string = generalized_poisson(k, mu, lambda_)
    f1 = ROOT.TF1("f", formula_string)
    result = f1.Eval(1000.)  
    return result

def median(data):
    """Calculate the median of a list of numbers."""
    n = len(data)
    sorted_data = sorted(data)
    middle = n // 2
    
    # If the length of data is even, return the average of the two middle numbers
    if n % 2 == 0:
        return (sorted_data[middle - 1] + sorted_data[middle]) / 2
    # If the length of data is odd, return the middle number
    else:
        return sorted_data[middle]

def iqr(data):
    """Calculate the Interquartile Range (IQR)."""
    n = len(data)
    sorted_data = sorted(data)
    middle = n // 2

    # If the length of data is even, split the data into two halves
    if n % 2 == 0:
        lower_half = sorted_data[:middle]
        upper_half = sorted_data[middle:]
    # If the length of data is odd, do not include the median in either half
    else:
        lower_half = sorted_data[:middle]
        upper_half = sorted_data[middle+1:]

    # Compute the medians of the lower and upper halves
    q1 = median(lower_half)
    q3 = median(upper_half)

    # Compute the IQR
    iqr_value = q3 - q1
    
    return iqr_value

def process_args():
    if len(sys.argv) < 4:
        print("Usage: python find_gaussian_peaks.py <input_file> <tree_name> <variable_name> <output_path>")
        sys.exit(1)
    input_file = os.path.abspath(sys.argv[1])
    tree_name = sys.argv[2]
    variable_name = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) == 5 else f"results/main_run_{input_file.split('/')[-1].split('_')[2]}"
    return input_file, tree_name, variable_name, output_path

def create_directories(output_path):
    for sub_dir in ["/csv", "/root"]:
        dir_path = output_path + sub_dir
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

def fetch_file_info(filename):
    pattern_name = r'(\w+)_run_(\w+)_ov_(\d+).00_sipmgr_(\d+)_(\w+)'
    name_match = re.match(pattern_name, filename)
    if name_match:
        return {
            'run': str(name_match.group(2)),
            'ov': int(name_match.group(3)),
            'channel': int(name_match.group(4)),
            'sipm_type': name_match.group(5)
        }
    else:
        return {}

def main():
    input_file, tree_name, variable_name, output_path = process_args()
    file_info = fetch_file_info(input_file.split("/")[-1])
    create_directories(output_path)

    ov = file_info.get('ov')
    run = file_info.get('run')
    channel = file_info.get('channel')
    sipm_type = file_info.get('sipm_type')
    num_bins = ov_ranges[ov]
    x_max = num_bins
    file1 = ROOT.TFile(input_file)
    tree = file1.Get(tree_name)
    n_entry = tree.GetEntries()
    combined_dict = defaultdict(list)
    output_file = ROOT.TFile(f"{output_path}/root/charge_fit_tile_ch{file_info.get('channel')}_ov{file_info.get('ov')}.root", "recreate") 
    sub_directory = output_file.mkdir(f"charge_fit_run_{run}")
    sub_directory.cd()
    for tile in range(16):
        tree.Draw(f"baselineQ_ch{tile}>>histogram{tile}")
        histogram = ROOT.gPad.GetPrimitive(f"histogram{tile}")
        baseline = histogram.GetMean()
        baseline_res = histogram.GetRMS()
        
        bin1 = histogram.GetXaxis().FindBin(- 3 * baseline_res);
        bin2 = histogram.GetXaxis().FindBin(3 * baseline_res);
        integral = histogram.Integral(bin1, bin2) / 0.9973;
    
        hist = ROOT.TH1F("hist","hist", int(num_bins), baseline - baseline_res * 5, baseline + float(x_max))
        original_bin_width = (float(x_max) + baseline_res * 5) / num_bins
        tree.Draw("{}>>hist".format(f'{variable_name}_ch{tile}'))
        # Print the final result
        #canvas_peaks = ROOT.TCanvas("c1","c1", 1200, 800)
        spectrum = ROOT.TSpectrum()
        n_peaks = spectrum.Search(hist, 0.2 , "", 0.05)
        #canvas_peaks.SetName(f"peak_finding_tile_{tile}_ch_{channel}_ov_{ov}")
        #canvas_peaks.Write()
        peaks_tspectrum = []
        for i in range(n_peaks):
            peaks_tspectrum.append(float(spectrum.GetPositionX()[i]))
        peaks_tspectrum.sort()
        if len(peaks_tspectrum) >= 2 and peaks_tspectrum[1] - peaks_tspectrum[0] > 8:
            distance = peaks_tspectrum[1] - peaks_tspectrum[0]
        else:
            distance = 10
        sigQ_datalist = []
        for entry in tree:
            branch_value = float(getattr(entry, f'{variable_name}_ch{tile}'))
            sigQ_datalist.append(branch_value)
        IQR = iqr(sigQ_datalist)#stats.iqr(sigQ_datalist, rng=(25, 75), scale="raw", nan_policy="omit")
        #IQR = calculateIQR(hist)  # Follow the Freedman-Diaconis rule for the further binning of the hist
        FD_bin_width = 2 * IQR * np.power(n_entry, -1/3) /2 
        sigQ = RooRealVar("sigQ", "sigQ", baseline - baseline_res * 5, baseline + float(x_max))
        #data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), ROOT.RooFit.Import(hist))
        data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), hist)
        # Define the parameters of the distribution
        lambda_ = RooRealVar("lambda", "lambda", 0.2, 0.005, 0.8)
        mu = RooRealVar("mu", "mu", 2, 0.1, 5)
        sigma0 = RooRealVar("sigma0", "sigma0", 5,1,10)
        sigmak = RooRealVar("sigmak", "sigmak", 2,0.01,10)
        A = RooRealVar("A", "A", 2,0.01,10)
        B = RooRealVar("B", "B", 2,0.01,10)
        C = RooRealVar("C", "C", 2,0.01,10)
        ped = RooRealVar("ped", "ped", baseline, baseline - baseline_res *2, baseline + baseline_res * 2)
        gain = RooRealVar("gain", "gain", distance, distance *0.8, distance *1.2)
        
        for i in range(1,17):
            #globals()[f'sigma{i}'] = RooRealVar(f"sigma{i}", "sigma{i}", 5,1,10)
            #globals()[f'sigma{i}'] = RooFormulaVar(f"sigma{i}", f"TMath::Sqrt( pow(sigma0, 2) + pow({i}, 2) * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
            globals()[f'sigma{i}'] = RooFormulaVar(f"sigma{i}", f"TMath::Sqrt( pow(sigma0, 2) + (A + B * {i} + C * {i} * {i}) * pow(sigmak, 2))", RooArgList(sigma0,sigmak, A, B, C))
        # Define the Gaussian PDF
        
        n_param = 9 # mu, lambda_, gain, ped, sigma0, sigmak, A, B, C
        argList = RooArgList(lambda_, mu, sigQ, ped, gain, sigma0)
        for i in range(1, ov_peaks[ov] + 1):
            argList.add(globals()[f"sigma{i}"])
        formula = prepare_GP_pdf(ov_peaks[ov])
          
        poisson_gen = RooGenericPdf("poisson_gen", formula, argList)
        # Specify the optimization algorithm
        minimizer = ROOT.RooMinimizer(poisson_gen.createNLL(data))
        minimizer.setMinimizerType("Minuit2")  # Choose Minuit2 as the optimizer
        
        # Perform the fit
        result = minimizer.fit("")
        mu_value = mu.getVal()
        lambda_value = lambda_.getVal()
        max_peak_to_fit = 0
        for k in range(1, 17):
            if calculate_GP(k, mu_value, lambda_value) > 0.002:
                max_peak_to_fit += 1
            else:
                break
        min_peak_to_fit = 0
        for k in range(1, 17):
            if calculate_GP(k, mu_value, lambda_value) > 0.05:
                min_peak_to_fit += 1
            else:
                break
        min_peak_to_fit = min(min_peak_to_fit, n_peaks)
        centers = []
        for k in range(min_peak_to_fit):
            if k == 0:
                sigmak_value = sigma0.getVal()
            else:
                sigmak_value = globals()[f'sigma{k}'].getVal()
            # Define the range for the fit
            lower_bound = ped.getVal() + k * gain.getVal() - 3 * sigmak_value
            upper_bound = ped.getVal() + k * gain.getVal() + 3 * sigmak_value
            # Define the variables for the fit
            mean_tmp = RooRealVar("mean_tmp", "mean_tmp", ped.getVal() + k * gain.getVal(), lower_bound, upper_bound)
            sigma_tmp = RooRealVar("sigma_tmp", "sigma_tmp", sigmak_value, sigmak_value * 0.5, sigmak_value * 1.5 )
            
            # Define the Gaussian PDF
            gaussian = RooGaussian("gaussian", "gaussian", sigQ, mean_tmp, sigma_tmp)
            
            # Fit the histogram in the specified range
            gaussian.fitTo(data, ROOT.RooFit.Range(lower_bound, upper_bound))
            
            # 3. Extract the mean (center) of each Gaussian
            centers.append(mean_tmp.getVal())
        gains = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
        if len(gains) != 0:
            average_gain = sum(gains) / len(gains)
        else:
            average_gain = 0
        new_x_min = ped.getVal() - sigma0.getVal() * 5
        new_x_max = ped.getVal() + globals()[f'sigma{max_peak_to_fit}'].getVal() * 3 + max_peak_to_fit * gain.getVal()
        new_Nbins = (new_x_max - new_x_min) / FD_bin_width
        
        new_hist = ROOT.TH1F("new_hist","new_hist", int(new_Nbins) + 1, new_x_min, new_x_max)
        tree.Draw("{}>>new_hist".format(f'{variable_name}_ch{tile}'))
        sigQ.setRange("newRange", new_x_min, new_x_max)
        new_data = ROOT.RooDataHist("new_data", "new_data", ROOT.RooArgSet(sigQ), new_hist)
        # Specify the optimization algorithm
        new_minimizer = ROOT.RooMinimizer(poisson_gen.createNLL(new_data))
        new_minimizer.setMinimizerType("Minuit2")  # Choose Minuit2 as the optimizer
        
        # Perform the fit
        result = new_minimizer.fit("")
        ## Create a chi-squared variable from the pdf and the data
        chi2 = ROOT.RooChi2Var("chi2", "chi2", poisson_gen, new_data)
        # Get the chi-squared value
        chi2_val = chi2.getVal()
        # Get number of degrees of freedom
        ndf = new_data.numEntries() - n_param #result.floatParsFinal().getSize()
        # Calculate chi-squared per degree of freedom
        chi2_ndf = chi2_val / ndf
        
        hist_sig = ROOT.TH1F("hists","hists", 10000, - baseline_res * 5,  10000)
        hist_bkg = ROOT.TH1F("histb","histb", 10000, - baseline_res * 5,  10000)
        tree.Draw("{}>>hists".format(f'{variable_name}_ch{tile} - {baseline}'))
        tree.Draw("{}>>histb".format(f'baselineQ_ch{tile} - {baseline}'))
        mean_sig = hist_sig.GetMean()
        stderr_sig = hist_sig.GetRMS()
        mean_bkg = hist_bkg.GetMean()
        print(mean_sig, mean_bkg)
        def error_for_ap(mean, lambda_, gain, mu, lambda_error, gain_error, mu_error):
            partial_lambda = -mean / (gain * mu)
            partial_gain = -mean * (1 - lambda_) / (gain**2 * mu)
            partial_mu = -mean * (1 - lambda_) / (gain * mu**2)
        
            dap = np.sqrt((partial_lambda * lambda_error)**2 + 
                            (partial_gain * gain_error)**2 + 
                            (partial_mu * mu_error)**2)
            return dap
        gain_value = gain.getVal() #if ov <= 2 else average_gain 
        gain_error = gain.getError()
        
        mu_val = mu.getVal()
        lambda_val = lambda_.getVal()
        mu_err = mu.getError()
        lambda_err = lambda_.getError()
        
        enf_data = float(mu_val * stderr_sig **2 / mean_sig**2)
        enf_data_err = float(mu_err * stderr_sig **2 / mean_sig**2)
        enf_GP = float(1 / (1-lambda_val))
        enf_GP_err = abs(1 / (1 - lambda_val)**2 * lambda_err)
        
        res_data = float(stderr_sig / mean_sig)
        res_GP = float(1 / np.sqrt(mu_val * (1 - lambda_val)))
        f_val = mu_val * (1 - lambda_val)
        delta_f = np.sqrt((1 - lambda_val)**2 * mu_err**2 + (-mu_val * lambda_err)**2)
        res_GP_err = abs(-0.5 / f_val**(1.5) * delta_f)
        ap = (enf_data - enf_GP) / mu_val
        rob_gain = float(stderr_sig ** 2 / mean_sig / enf_data / enf_GP)
        rob_gain_err = np.sqrt(
    (-stderr_sig**2 * enf_data_err / (mean_sig * enf_data**2 * enf_GP))**2 +
    (-stderr_sig**2 * enf_GP_err / (mean_sig * enf_data * enf_GP**2))**2
)
        
        ap_err = np.sqrt(
            (1 / mu_val * enf_data_err)**2 +
            (-1 / mu_val * enf_GP_err)**2 +
            (-1 * (enf_data - enf_GP) / mu_val**2 * mu_err)**2
        )
        
        fit_info = {
            'mu': float(mu_val - mean_bkg),
            'mu_dcr': float(mean_bkg / gain_value),  # Changed gain.getVal() to gain_value
            'mu_err': mu_err,
            'lambda': lambda_val,
            'lambda_err': lambda_err,
            'ap': float(ap),
            'ap_err': float(ap_err),  # Placeholder for ap_err
            'mean': float(mean_sig),
            'stderr': float(stderr_sig),
            'enf_GP': enf_GP,
            'enf_GP_err': float(enf_GP_err),
            'enf_data': enf_data,
            'enf_data_err': float(enf_data_err),
            'res_data': res_data,
            'res_GP': res_GP,
            'res_GP_err': float(res_GP_err),
            'ped': float(ped.getVal()),
            'ped_err': float(ped.getError()),
            'gain': float(gain_value),
            'rob_gain': float(rob_gain),
            'rob_gain_err': float(rob_gain_err),
            'avg_gain': float(average_gain),
            'gain_err': float(gain_error),  # Changed gain.getError() to gain_error
            'chi2': chi2_ndf,
            'sigma0': float(sigma0.getVal()),
            'sigmak': float(sigmak.getVal()),
            'a': float(A.getVal()),
            'b': float(B.getVal()),
            'c': float(C.getVal()),
            'n_peaks': n_peaks,
        }

        fit_info['events'] = n_entry
        fit_info['run'] = run
        fit_info['vol'] = ov
        fit_info['ch'] = channel
        fit_info['pos'] = tile
        fit_info['bl_rms'] = float(baseline_res / 45)
        fit_info['bl'] = float(baseline / 45)
        print(distance, gain.getVal())
        print(mu.getVal(), lambda_.getVal())
        print(len(peaks_tspectrum))
        for key, value in fit_info.items():
            combined_dict[key].append(value)
        # Plot the data and the fit
        canvas = ROOT.TCanvas("c1","c1", 1200, 800)
        #if tile == 0:
        #    canvas.Print(f"{output_path}/pdf/charge_fit_tile_ch{channel}_ov{ov}.pdf[")
    
        print("FD bin width:", FD_bin_width, " original bin width:", original_bin_width )
        print("finer fitted gain:", average_gain)
        # Divide the canvas into two asymmetric pads
        pad1 =ROOT.TPad("pad1","This is pad1",0.05,0.05,0.72,0.97);
        pad2 = ROOT.TPad("pad2","This is pad2",0.72,0.05,0.98,0.97);
        pad1.Draw()
        pad2.Draw()
        pad1.cd()
        frame = sigQ.frame()
        frame.SetXTitle("Charge")
        frame.SetYTitle("Events")
        frame.SetTitle(f"Charge spectrum fit of (run{run} ch{channel} tile{tile} ov{ov}V)")
        new_data.plotOn(frame)
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
        param_box.AddText(f"#sigma0 = {sigma0.getVal():.3f} ")
        param_box.AddText(f"#sigma1 = {sigma1.getVal():.3f} ")
        param_box.AddText(f"#sigma2 = {sigma2.getVal():.3f} ")
        param_box.AddText(f"#sigma3 = {sigma3.getVal():.3f} ")
        param_box.AddText(f"#sigma4 = {sigma4.getVal():.3f} ")
        param_box.AddText(f"#sigma5 = {sigma5.getVal():.3f} ")
        param_box.AddText(f"#chi2/NDF = {chi2_ndf:.3f}")
        param_box.Draw("same")
        #canvas.Print(f"{output_path}/pdf/charge_fit_tile_ch{channel}_ov{ov}.pdf")
        canvas.SetName(f"charge_spectrum_tile_{tile}_ch_{channel}_ov_{ov}")
        canvas.Write()
        for k in range(1, 17):
            if calculate_GP(k, mu_value, lambda_value) > 0.005:
                print(k, calculate_GP(k, mu_value, lambda_value))
                max_peak_to_fit += 1
            else:
                break
        print("max peak:", k)
    
    #canvas.Print(f"{output_path}/pdf/charge_fit_tile_ch{channel}_ov{ov}.pdf]")
    output_file.Close()

    df = pd.DataFrame(combined_dict)
    df.to_csv(f"{output_path}/csv/charge_fit_tile_ch{file_info.get('channel')}_ov{file_info.get('ov')}.csv", index=False)

if __name__ == "__main__":
    main()

