import os
import sys
import re
import math
from time import time
import numpy as np
import pandas as pd
from compound_pdf import compound_pdf_str
from collections import defaultdict
import ROOT
from ROOT import (TMath, RooRealVar, RooGenericPdf, RooArgSet, RooDataHist, 
                  RooFit, TCanvas, TH1F, TRandom3, RooFormulaVar, RooGaussian,
                  RooArgList)
import multiprocessing

cpu_count = multiprocessing.cpu_count()

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

def standard_error_of_average_gain(gains):
    n = len(gains)
    average_gain = sum(gains) / n
    variance = sum([(gain - average_gain) ** 2 for gain in gains]) / (n - 1)
    standard_deviation = variance ** 0.5
    standard_error = standard_deviation / n ** 0.5
    return standard_error

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
def map_to_nearest_smaller_ap(lambda_input, enf_residual_input):
    if lambda_input < 0.002 or lambda_input > 0.998:
        return 0.
    if enf_residual_input < 0.0001:
        return 0.
    # Load the data
    data = pd.read_csv("/junofs/users/wanghanwen/ap_table.csv")
    
    # Find the closest lambda value in the data
    lambda_closest = data['lambda'].iloc[(data['lambda'] - lambda_input).abs().argsort()[:1]].values[0]
    
    # Filter the data for the closest lambda
    data_lambda_filtered = data[data['lambda'] == lambda_closest]
    
    # If the given enf_residual exceeds the range, use the row with the maximum 'enf_residual'
    if enf_residual_input > data_lambda_filtered['enf_residual'].max():
        ap_value = data_lambda_filtered.loc[data_lambda_filtered['enf_residual'].idxmax(), 'ap']
    else:
        # Find the two closest enf_residual values in the filtered data
        closest_indices = data_lambda_filtered['enf_residual'].sub(enf_residual_input).abs().nsmallest(4).index
        closest_data = data_lambda_filtered.loc[closest_indices]
        
        # Get the 'ap' value corresponding to the smaller of the two closest 'enf_residual' values
        ap_value = closest_data['ap'].min()
    
    return ap_value


def process_args():
    if len(sys.argv) < 5:
        print("Usage: python find_gaussian_peaks.py <input_file> <tree_name> <variable_name> <output_path>")
        sys.exit(1)
    if len(sys.argv) == 5:
        input_file = os.path.abspath(sys.argv[1])
        tree_name = sys.argv[2]
        variable_name = sys.argv[3]
        output_path = sys.argv[4] 
        return [input_file, tree_name, variable_name, output_path]
    elif len(sys.argv) == 6:
        input_file = os.path.abspath(sys.argv[1])
        tree_name = sys.argv[2]
        variable_name = sys.argv[3]
        output_path = sys.argv[4] 
        csv_path = sys.argv[5] 
        return [input_file, tree_name, variable_name, output_path, csv_path]

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
    if len(process_args()) == 4:
        input_file, tree_name, variable_name, output_path = process_args()
    elif len( process_args()) == 5:
        input_file, tree_name, variable_name, output_path, csv_path = process_args()
   
    fix_parameters = False 
    if len( process_args()) == 5:
        file_path = csv_path
        if os.path.isfile(file_path) and file_path.endswith(".csv"):
            fix_parameters = True
            print(f"{file_path} csv file exists!")
        else:
            print(f"{file_path} csv file does not exist!")
    if fix_parameters:
        df_params = pd.read_csv(csv_path)
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
    if os.path.isfile(f"{output_path}/csv/charge_fit_tile_ch{file_info.get('channel')}_ov{file_info.get('ov')}.csv"):
        os.system(f"rm {output_path}/csv/charge_fit_tile_ch{file_info.get('channel')}_ov{file_info.get('ov')}.csv")
    if os.path.isfile(f"{output_path}/root/charge_fit_tile_ch{file_info.get('channel')}_ov{file_info.get('ov')}.root"):
        os.system(f"rm {output_path}/root/charge_fit_tile_ch{file_info.get('channel')}_ov{file_info.get('ov')}.root")
    ROOT.TVirtualFitter.SetDefaultFitter("Minuit2")
    for tile in range(16):
    #for tile in range(6,7):

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
        spectrum = ROOT.TSpectrum()
        n_peaks = spectrum.Search(hist, 0.2 , "", 0.05)
        if baseline >= 5000 or baseline <= -5000 or baseline_res == 0 or baseline_res > 300:
            fit_info = {
                'cpu_num' : int(cpu_count),
                'prefit_status':   -2, 
                'prefit_cpuTime':  0, 
                'mufit_status':     0,
                'mufit_cpuTime':    0,
                'finalfit_status':  0,
                'finalfit_cpuTime': 0,
                'iterfit_cpuTime':  0,
                'iterfit_num':      0,
                'iterfit_num_success': 0,
                'mu': 0.,
                'mu_err': 0.,
                'lambda': 0.,
                'lambda_err': 0.,
                'alpha': 0.,
                'alpha_err': 0.,  # Placeholder for ap_err
                'mean': float(baseline),
                'stderr': float(baseline_res),
                'enf_GP': 0.,
                'enf_GP_err': 0,
                'enf_data': 0,
                'enf_data_err': 0,
                'res_data': 0,
                'res_GP': 0,
                'res_GP_err': 0,
                'ped': 0,
                'ped_err': 0,
                'gain': 0,
                'gain_err': 0,  # Changed gain.getError() to gain_error
                'ap_gain': 0,
                'ap_gain_err': 0,  # Changed gain.getError() to gain_error
                'rob_gain': 0,
                'rob_gain_err': 0,
                'chi2': 0,
                'n_peaks_tofit': 0, 
                'lower_edge': 0,
                'upper_edge': 0,
                'nbins': 0,
                'fd_bin_width': 0,
                'bin_width': 0,
                'charge_fit_ndf': 0,
                'sigma0': 0,
                'sigma0_err': 0,#float(sigma0.getError()),
                'sigmaAp': 0,
                'sigmaAp_err': 0,#float(sigma0.getError()),
                'sigmak': 0,#float(sigma0.getError()),
                'sigmak_err': 0,#float(sigma0.getError()),
                'n_peaks': 0,
            }

            fit_info['events'] = n_entry
            fit_info['run'] = int(run)
            fit_info['vol'] = ov
            fit_info['ch'] = channel
            fit_info['pos'] = tile
            fit_info['bl_rms'] = float(baseline_res)
            fit_info['bl'] = float(baseline)
            for key, value in fit_info.items():
                combined_dict[key].append(value)
            continue
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
        sigQ_baseline = RooRealVar("sigQ_bl", "sigQ_bl", baseline - baseline_res * 10, baseline + baseline_res * 10)
        #data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), ROOT.RooFit.Import(hist))
        data = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(sigQ), hist)
        data_baseline = ROOT.RooDataHist("data_baseline", "data_baseline", ROOT.RooArgSet(sigQ_baseline), histogram)
        sigma0 = RooRealVar("sigma0", "sigma0", 5,1,20)
        
        ped = RooRealVar("ped", "ped", baseline, baseline - baseline_res *2, baseline + baseline_res * 2)
        gauss_baseline = RooGaussian("gauss_bl","gauss_bl", sigQ, ped, sigma0)
        gauss_baseline.fitTo(data_baseline)
        bl_sigma = sigma0.getVal()
        bl_mean = ped.getVal()
        #sigma0.setConstant(ROOT.kTRUE)
        #ped.setConstant(ROOT.kTRUE)
        # Define the parameters of the distribution
        lambda_ = RooRealVar("lambda", "lambda", 0.2, 0.005, 0.8)
        mu = RooRealVar("mu", "mu", 2, 0.1, 5)
        sigmak = RooRealVar("sigmak", "sigmak", 2,0.01,10)
        #beta = RooRealVar("beta", "beta", 4.0)
        # Initialize the variables
        
        # Define the RooRealVars
        
        # Create the RooFormulaVar
        #beta = RooFormulaVar("beta", "Beta Formula", beta_formula, RooArgList(tau_ap, t_gate_var, tau_R_var))
        sigmaAp = RooRealVar("sigmaAp", "sigmaAp",0)# beta_formula, RooArgList(tau_ap, t_gate_var, tau_R_var))

        #beta = RooRealVar("beta", "beta", 0.01,0.0001,10)
        alpha = RooRealVar("alpha", "alpha", 0.01, 0.0, 0.3)
        A = RooRealVar("A", "A", 0,-10,10)
        B = RooRealVar("B", "B", 0,-10,10)
        C = RooRealVar("C", "C", 0,-10,10)
        shift = RooRealVar("shift", "shift", 0, 0, float(ov+1)*3.5)

        if fix_parameters and n_peaks < 3:
            filtered_df_params = df_params[(df_params['run'] == int(run)) & (df_params['pos'] == tile) & (df_params['ch'] == int(channel)) & (df_params['vol'] == int(ov))]

            if len(filtered_df_params['gain'].tolist()) != 0:
                gain_value = filtered_df_params.head(1)['fit_gain'].values[0]
                gain_value = filtered_df_params.head(1)['prefit_gain'].values[0] if gain_value == 0 else gain_value
                gain_value = filtered_df_params.head(1)['rob_gain'].values[0] if gain_value == 0 else gain_value
                gain_err = filtered_df_params.head(1)['fit_gain_err'].values[0]
                gain_err = filtered_df_params.head(1)['prefit_gain_err'].values[0] if gain_err == 0 else gain_err
                gain_err = filtered_df_params.head(1)['rob_gain_err'].values[0] if gain_err == 0 else gain_err
            gain = RooRealVar("gain", "gain", gain_value)
            gain.setError(gain_err)
            gain.setConstant(ROOT.kTRUE)
        else:
            gain = RooRealVar("gain", "gain", distance, distance *0.5, distance *1.5)

        for n in range(1,18):
            for i in range(0,n+1):
            #globals()[f'sigma{i}'] = RooRealVar(f"sigma{i}", "sigma{i}", 5,1,15)
            #globals()[f'sigma{i}'] = RooFormulaVar(f"sigma{i}", f"TMath::Sqrt( pow(sigma0, 2) + pow({i}, 2) * pow(sigmak, 2))", RooArgList(sigma0,sigmak))
            #globals()[f'sigma{i}'] = RooFormulaVar(f"sigma{i}", f"TMath::Sqrt( pow(sigma0, 2) + (A + B * {i} + C * {i} * {i}) * pow(sigmak, 2))", RooArgList(sigma0,sigmak, A, B, C))
                #globals()[f'sigma_{n}_{i}'] = RooFormulaVar(f"sigma_{n}_{i}", f"TMath::Sqrt( pow(sigma0, 2) +  {n} * pow(sigmak, 2) + {i} * pow(sigmaAp, 2))", RooArgList(sigma0,sigmak, sigmaAp))
                #globals()[f'sigma_{n}_{i}'] = RooFormulaVar(f"sigma_{n}_{i}", f"TMath::Sqrt( pow(sigma0, 2) + (A + B * {i} + C * {i} * {i}) * pow(sigmak, 2))", RooArgList(sigma0,sigmak, A, B, C))
                globals()[f'sigma_{n}_{i}'] = RooFormulaVar(f"sigma_{n}_{i}", f"TMath::Sqrt( pow(sigma0, 2) + {n} * pow(sigmak, 2) + {i} * pow(sigmaAp, 2) )", RooArgList(sigma0,sigmak, sigmaAp))
        # Define the Gaussian PDF
        argList = RooArgList(lambda_, mu, sigQ, ped, gain,shift,alpha,sigma0, sigmak, sigmaAp)
        for n in range(1, 18):
            for i in range(0,n + 1):
                argList.add(globals()[f"sigma_{n}_{i}"])
          
        formula = compound_pdf_str
        poisson_gen = RooGenericPdf("poisson_gen", formula, argList)
        # Specify the optimization algorithm
        #minimizer = ROOT.RooMinimizer(poisson_gen.createNLL(data))
        #minimizer.setMinimizerType("Minuit2")  # Choose Minuit2 as the optimizer
        
        # Perform the fit
        #result = minimizer.fit("")
        alpha.setVal(0)
        alpha.setConstant(ROOT.kTRUE)
        start_time1 = time()
        result1 = poisson_gen.fitTo(data, ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save(), ROOT.RooFit.NumCPU(cpu_count, 0))
        end_time1 = time()
        prefit_cpuTime = end_time1 - start_time1
        prefit_status =  result1.status() 
        #baseline_events = hist.Integral(hist.FindFixBin(3680), hist.FindFixBin(3725))
        #calculated_mu = - np.log(baseline_events / hist.Integral())
        #mu.setVal(calculated_mu)
        initialParams = poisson_gen.getParameters(data).snapshot()
        #poisson_gen.fitTo(data, ROOT.RooFit.SumW2Error(ROOT.kTRUE),ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save(), ROOT.RooFit.Strategy(2), ROOT.RooFit.InitialHesse(ROOT.kTRUE))
        mu_value = mu.getVal()
        lambda_value = lambda_.getVal()
        max_peak_to_fit = 0
        for k in range(1, 16):
            if calculate_GP(k, mu_value, lambda_value) > 0.002:
                max_peak_to_fit += 1
            else:
                break
        new_x_min = ped.getVal() - sigma0.getVal() * 5
        new_x_max = ped.getVal() + globals()[f'sigma_{max_peak_to_fit}_0'].getVal() * 5 + max_peak_to_fit * gain.getVal()
        new_Nbins = (new_x_max - new_x_min) / (FD_bin_width )
        
        new_hist = ROOT.TH1F("new_hist","new_hist", int(new_Nbins) + 1, new_x_min, new_x_max)
        tree.Draw("{}>>new_hist".format(f'{variable_name}_ch{tile}'))
        sigQ.setRange("newRange", new_x_min, new_x_max)
        new_data = ROOT.RooDataHist("new_data", "new_data", ROOT.RooArgSet(sigQ), new_hist)
        # Specify the optimization algorithm
        #new_minimizer = ROOT.RooMinimizer(poisson_gen.createNLL(new_data))
        #new_minimizer.setMinimizerType("Minuit2")  # Choose Minuit2 as the optimizer
        params = poisson_gen.getParameters(data)
        params.assignValueOnly(initialParams)
        start_time2 = time()
        result2 = poisson_gen.fitTo(new_data,ROOT.RooFit.SumW2Error(ROOT.kTRUE),ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save(), ROOT.RooFit.Strategy(2), ROOT.RooFit.NumCPU(cpu_count, 0))
        end_time2 = time()
        mufit_cpuTime = end_time2 - start_time2
        mufit_status =  result2.status() 
        mu.setConstant(ROOT.kTRUE)
        alpha.setConstant(ROOT.kFALSE)
        initial_FD_bin_width = FD_bin_width
        best_bin_width = FD_bin_width
        best_chi2 = np.inf
        iterfit_cpuTime = []
        iterfit_status = []
        while FD_bin_width > 0.5 * initial_FD_bin_width:
            FD_bin_width -= 0.05 * initial_FD_bin_width
            new_Nbins = (new_x_max - new_x_min) / FD_bin_width
            
            new_hist = ROOT.TH1F("new_hist","new_hist", int(new_Nbins) + 1, new_x_min, new_x_max)
            tree.Draw("{}>>new_hist".format(f'{variable_name}_ch{tile}'))
            new_data = ROOT.RooDataHist("new_data", "new_data", ROOT.RooArgSet(sigQ), new_hist)
            start_time3 = time()
            result3 = poisson_gen.fitTo(new_data, ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save(), ROOT.RooFit.Strategy(2), ROOT.RooFit.NumCPU(cpu_count, 0))
            end_time3 = time()
            iterfit_cpuTime.append(end_time3 - start_time3)
            iterfit_status.append(result3.status())
            shift_max = shift.getMax()
            shift_min = shift.getMin()
            threshold = 0.01 * (shift_max - shift_min)
            good_ap_fit = shift.getVal() - shift_min > threshold or shift_max - shift.getVal() > threshold 
            chi2 = ROOT.RooChi2Var("chi2", "chi2", poisson_gen, new_data)
            ## Get the chi-squared value
            tmp_chi2_val = chi2.getVal()
            if tmp_chi2_val < best_chi2 and result3.status() != 4 and good_ap_fit:
                best_bin_width = FD_bin_width
                best_chi2 = tmp_chi2_val
        iterfit_cpuTime = np.array(iterfit_cpuTime)
        iterfit_status = np.array(iterfit_status)
        iterfit_num = len(iterfit_cpuTime)
        iterfit_average_cpuTime = np.mean(iterfit_cpuTime)
        iterfit_num_success = np.sum(iterfit_status == 0)
        

        new_Nbins = (new_x_max - new_x_min) / best_bin_width
        
        new_hist = ROOT.TH1F("new_hist","new_hist", int(new_Nbins) + 1, new_x_min, new_x_max)
        tree.Draw("{}>>new_hist".format(f'{variable_name}_ch{tile}'))
        new_data = ROOT.RooDataHist("new_data", "new_data", ROOT.RooArgSet(sigQ), new_hist)
        start_time4 = time()
        result4 = poisson_gen.fitTo(new_data, ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save(), ROOT.RooFit.Strategy(2), ROOT.RooFit.NumCPU(cpu_count, 0))
        end_time4 = time()
        finalfit_cpuTime = end_time4 - start_time4
        finalfit_status = result4.status()
        
            

        ## Create a chi-squared variable from the pdf and the data
        chi2 = ROOT.RooChi2Var("chi2", "chi2", poisson_gen, new_data)
        # Get the chi-squared value
        chi2_val = chi2.getVal()
        # Get number of degrees of freedom
        ndf = int(new_Nbins) - result2.floatParsFinal().getSize()
        # Calculate chi-squared per degree of freedom
        chi2_ndf = chi2_val / ndf
        
        hist_sig = ROOT.TH1F("hists","hists", 10000, - baseline_res * 5,  10000)
        tree.Draw("{}>>hists".format(f'{variable_name}_ch{tile} - {bl_mean}'))
        mean_sig = hist_sig.GetMean()
        stderr_sig = hist_sig.GetRMS()

        gain_value = gain.getVal() #if ov <= 2 else average_gain 
        gain_error = gain.getError()
        
        mu_val = mu.getVal()
        lambda_val = lambda_.getVal()
        mu_err = mu.getError()
        lambda_err = lambda_.getError()
        alpha_val = alpha.getVal()
        alpha_err = alpha.getError()
        #tau_ap_val = tau_ap.getVal()
        
        enf_data = float(mu_val * stderr_sig **2 / mean_sig**2)
        enf_data_err = float(mu_err * stderr_sig **2 / mean_sig**2)
        enf_GP = float(1 / (1-lambda_val))
        enf_GP_err = abs(1 / (1 - lambda_val)**2 * lambda_err)
        
        res_data = float(stderr_sig / mean_sig)
        res_GP = float(1 / np.sqrt(mu_val * (1 - lambda_val)))
        f_val = mu_val * (1 - lambda_val)
        delta_f = np.sqrt((1 - lambda_val)**2 * mu_err**2 + (-mu_val * lambda_err)**2)
        res_GP_err = abs(-0.5 / f_val**(1.5) * delta_f)
        f_corr = 1 - sigma0.getVal() **2 / stderr_sig ** 2
        enf_res = (enf_data * f_corr - enf_GP) / mu_val 
        rob_gain = float(mean_sig / (mu_val/(1-lambda_val)))
        rob_gain_err = math.sqrt((mean_sig**2 * mu_err**2 * (1 - lambda_val)**2) / (mu_val**4) + 
                         (mean_sig**2 * lambda_err**2) / (mu_val**2))
        fit_info = {
            'cpu_num' : int(cpu_count),
            'prefit_status': int(prefit_status),
            'prefit_cpuTime': float(prefit_cpuTime),
            'mufit_status': int(mufit_status),
            'mufit_cpuTime': float(mufit_cpuTime),
            'finalfit_status': int(finalfit_status),
            'finalfit_cpuTime': float(finalfit_cpuTime),
            'iterfit_cpuTime': float(iterfit_average_cpuTime),
            'iterfit_num': int(iterfit_num),
            'iterfit_num_success': int(iterfit_num_success),
            'mu': float(mu_val),
            'mu_err': mu_err,
            'lambda': lambda_val,
            'lambda_err': lambda_err,
            'alpha': float(alpha_val),
            'alpha_err': float(alpha_err),  # Placeholder for ap_err
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
            'ap_gain': float(shift.getVal()),
            'ap_gain_err': float(shift.getError()),
            'rob_gain': float(rob_gain),
            'rob_gain_err': float(rob_gain_err),
            'gain_err': float(gain_error),  # Changed gain.getError() to gain_error
            'chi2': chi2_ndf,
            'charge_fit_ndf': float(ndf),
            'n_peaks_tofit': float(max_peak_to_fit),
            'lower_edge': float(new_x_min),
            'upper_edge': float(new_x_max),
            'nbins': float(new_Nbins),
            'fd_bin_width': float(initial_FD_bin_width),
            'bin_width': float(FD_bin_width),
            'sigmaAp': float(sigmaAp.getVal()),
            'sigmaAp_err': float(sigmaAp.getError()),
            'sigma0': float(sigma0.getVal()),
            'sigma0_err': float(sigma0.getError()),
            'sigmak': float(sigmak.getVal()),
            'sigmak_err': float(sigma0.getError()),
            'n_peaks': n_peaks,
        }

        fit_info['events'] = n_entry
        fit_info['run'] = int(run)
        fit_info['vol'] = ov
        fit_info['ch'] = channel
        fit_info['pos'] = tile
        fit_info['bl_rms'] = float(bl_sigma)
        fit_info['bl'] = float(bl_mean)
        for key, value in fit_info.items():
            combined_dict[key].append(value)

    df = pd.DataFrame(combined_dict)
    df.to_csv(f"{output_path}/csv/charge_fit_tile_ch{file_info.get('channel')}_ov{file_info.get('ov')}.csv", index=False)

if __name__ == "__main__":
    main()

