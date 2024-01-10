import pandas as pd
import math
import random
import re
import os, sys
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from copy import deepcopy

import statsmodels.api as sm

special_runs = {425:48, 426:48, 427:48, 428:48, 429:50, 430:48, 431:48, 432:49, 433:49, 435:50, 436:47,}
def remove_outliers(x_list, y_list, y_err_list, threshold=2):
    if len(x_list) >= 3:
        # Add a constant (intercept term) to predictors
        X = sm.add_constant(np.array(x_list))
        y = np.array(y_list)

        # Fit the model
        model = sm.OLS(y, X)
        results = model.fit()

        # Calculate residuals
        residuals = results.resid

        # Standardize residuals
        standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

        # Identify outliers
        outliers = np.abs(standardized_residuals) > threshold
        #print(np.abs(standardized_residuals))
        #print("outliers", outliers)

        # Create new lists without outliers
        x_list_new = np.array(x_list)[~outliers].tolist()
        y_list_new = np.array(y_list)[~outliers].tolist()
        y_err_list_new = np.array(y_err_list)[~outliers].tolist()

        return x_list_new, y_list_new, y_err_list_new
    else:
        print("length of the array is less than 5, skipping the outliers removal")
        return x_list, y_list, y_err_list


def draw_linear_fit(fig, ax, x, x_with_bkv, y, y_err, intercept, slope, x_intercept_err, chi2):
    # convert input lists to numpy arrays
    x = np.array(x)
    xplus = np.array(x_with_bkv)
    y = np.array(y)
    y_err = np.array(y_err)

    # Define the fitted line
    fit = slope * xplus + intercept
    # Plot the data points with error bars
    plt.errorbar(x, y, yerr=y_err, fmt='o',markersize=2, color='black', label='Charge Data')

    # Plot the fitted line
    plt.plot(xplus, fit, color='red', label='Charge fit')
    # Set the limits for the twin y-axis
    plt.ylim([0, np.max(y) * 1.1])

    # Add labels and legend
    plt.xlabel('Voltage(V)')
    plt.ylabel(f'Charge')
    plt.legend()
    # Add chi-square value
    plt.text(0.05, 0.75, f'$\chi^2$/ndf: {chi2:.5f}', transform=plt.gca().transData, verticalalignment='top')
    plt.text(0.05, 0.65, f'Breakdown Voltage: {x_with_bkv[0]:.3f} $\pm$ {x_intercept_err:.3f}', transform=plt.gca().transData, verticalalignment='top')

def calculate_x_intercept_error(m, slope, m_err, slope_err):
    # Calculate the derivative of x_intercept with respect to m
    d_x_intercept_dm = -1 / slope

    # Calculate the derivative of x_intercept with respect to slope
    d_x_intercept_dslope = -m / (slope ** 2)

    # Calculate the partial derivatives of x_intercept with respect to m_err and slope_err
    d_x_intercept_dm_err = d_x_intercept_dm * m_err
    d_x_intercept_dslope_err = d_x_intercept_dslope * slope_err

    # Calculate the error in the x-intercept using error propagation formula
    x_intercept_err = math.sqrt(d_x_intercept_dm_err**2 + d_x_intercept_dslope_err**2)

    return x_intercept_err

def linear_fit_bootstrap(x_list, y_list, y_err_list, n_bootstrap=1000):
    # convert input lists to numpy arrays
    x = np.array(x_list)
    y = np.array(y_list)
    y_err = np.array(y_err_list)

    # perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # calculate x-intercept
    x_intercept = -intercept / slope
    residuals = y - (slope * x + intercept)
    squared_residuals = residuals ** 2
    var = np.mean(squared_residuals)
    # Calculate the sum of squared residuals (SSR)
    SSR = np.sum(squared_residuals)
    #print(SSR)
    degrees_of_freedom = len(x) - 2  # Number of data points minus number of fitted parameters
    reduced_chi2 = SSR/ var / degrees_of_freedom

    # bootstrap resampling
    bootstrap_slopes = []
    bootstrap_intercepts = []
    bootstrap_x_intercepts = []

    for _ in range(n_bootstrap):
        # create bootstrap sample by varying each element of y_list separately
        y_values = []
        for i in range(len(y)):
            rand = random.gauss(0, 1)
            y_values.append(y[i] + rand * y_err[i])
        bootstrap_y = np.array(y_values)

        # perform linear regression on the bootstrap sample
        bootstrap_slope, bootstrap_intercept, _, _, _ = stats.linregress(x, bootstrap_y)

        # calculate x-intercept for bootstrap sample
        bootstrap_x_intercept = -bootstrap_intercept / bootstrap_slope

        # store bootstrap results
        bootstrap_slopes.append(bootstrap_slope)
        bootstrap_intercepts.append(bootstrap_intercept)
        bootstrap_x_intercepts.append(bootstrap_x_intercept)

    # calculate uncertainties using bootstrap results
    slope_std_err = np.std(bootstrap_slopes)
    intercept_std_err = np.std(bootstrap_intercepts)
    x_intercept_std_err = np.std(bootstrap_x_intercepts)
    x_intercept_err = np.sqrt(x_intercept_std_err ** 2)

    return (
        slope,
        intercept,
        x_intercept,
        slope_std_err,
        intercept_std_err,
        x_intercept_err,
        SSR,
        degrees_of_freedom
    )


if len(sys.argv) < 3:
    print("Usage: python prepare_jobs <input_file> <output_dir>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
input_file =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
output_dir = os.path.abspath(output_tmp)
#/junofs/users/wanghanwen/main-runs/vbd/main_run_0063
name_short = output_dir.split("/")[-1]
components = name_short.split("_")
run = int(components[-1])
if run in special_runs.keys():
    init_vol = special_runs[run]
else:
    init_vol = 48

# Find a certain element based on other column values

if not os.path.isdir(output_dir + "/csv"):
    os.makedirs(output_dir + "/csv")

if not os.path.isdir(output_dir + "/pdf"):
    os.makedirs(output_dir + "/pdf")
# Read the CSV file into a pandas DataFrame
if not os.path.isdir(input_file):
    df = pd.read_csv(input_file)
else:
    all_data = []
    for filename in os.listdir(input_file):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_file, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)
    df = pd.concat(all_data, ignore_index=True)
    

# Find a certain element based on other column values
#channel = 1
type_ = 'tile'
#position = 0
run_type = 'main'
#voltage = 1
#peak = 0
var_dict = {"sigAmp":"Amplitude", "sigQ":"Charge"}
unit_dict = {"sigAmp":"Amp (mV)", "sigQ":"Q (pC)"}


# Get the dictionary of column names and their indexes
column_dict = {column: index for index, column in enumerate(df.columns)}

# Access the desired element


# Set up the plot
fig, ax = plt.subplots()
#vols = [i for i in range(2,7)]
for position in range(16):
    df_tmp = pd.DataFrame()
 
    vbd_dict = []
    ov_dict = []
    prefit_gain = []
    prefit_gain_err = []
    fitted_gain = []
    fitted_gain_err = []
    rob_vbd_dict = []
    vbd_err_dict = []
    rob_vbd_err_dict = []
    chi2_list = []
    ndf_list = []
    slope_list = []
    slope_err_list = []
    voltage_missing_list_all = []
    fit_failure_list_all = []
    large_chi2_list_all = []
    parameter_miss_list_all = []
    # Create a PDF merger object
    #pdf_merger = PdfMerger()
    for channel in range(1,17):

      
        # Create a twin y-axis on the right side

        gains = []
        gain_errors = []
        rob_gains = []
        rob_gain_errors = []
        vols = []
        rob_vols = []
        voltage_missing_list = []
        fit_failure_list = []
        large_chi2_list = []
        parameter_miss_list = []
        for voltage in range(1,7):
            #print(position, channel, voltage)
            filtered_df = df.loc[ (df['ch'] == channel) &
                            (df['pos'] == position) &(df['vol'] == voltage)]
            #chi2 = filtered_df.head(1)['chi2'].values[0]
            voltage_missing_list.append(0)
            fit_failure_list.append(0)
            large_chi2_list.append(0)
            parameter_miss_list.append(0)
            if len(filtered_df['finalfit_status'].tolist()) != 0:
                mean = filtered_df.head(1)['mean'].values[0] 
                stderr = filtered_df.head(1)['stderr'].values[0] 
                if abs(mean) < 10 and stderr < 10:
                    prefit_gain.append(0)
                    prefit_gain_err.append(0)
                    voltage_missing_list[-1] = 1
                    continue
                status = filtered_df.head(1)['finalfit_status'].values[0]
                if status != 0:
                    prefit_gain.append(0)
                    prefit_gain_err.append(0)
                    fit_failure_list[-1] = 1
                    continue
                charge_spectrum_fit_chi2 = filtered_df.head(1)['chi2'].values[0]
                if charge_spectrum_fit_chi2 > 2:
                    prefit_gain.append(0)
                    prefit_gain_err.append(0)
                    large_chi2_list[-1] = 1
                    continue
            else:
                prefit_gain.append(0)
                prefit_gain_err.append(0)
                parameter_miss_list[-1] = 1
                continue
            hasRobGain = True
            if len(filtered_df['rob_gain'].tolist()) != 0:
                rob_gain = filtered_df.head(1)['rob_gain'].values[0]
                rob_gain_err = filtered_df.head(1)['rob_gain_err'].values[0]
            if len(filtered_df['gain'].tolist()) != 0:
                gain = filtered_df.head(1)['gain'].values[0]
                gain_err = filtered_df.head(1)['gain_err'].values[0]
                prefit_gain.append(gain)
                prefit_gain_err.append(gain_err)
            else:
                prefit_gain.append(0)
                prefit_gain_err.append(0)
            print(position, channel, voltage, gain, gain_err)
            gains.append(gain)
            gain_errors.append(gain_err)
            vols.append(voltage + init_vol)
            
            rob_gains.append(rob_gain)
            rob_gain_errors.append(rob_gain_err)
            rob_vols.append(voltage + init_vol)

        vols, gains, gain_errors = remove_outliers(vols, gains, gain_errors)
        rob_vols, rob_gains, rob_gain_errors = remove_outliers(rob_vols, rob_gains, rob_gain_errors)
        #slope, x_intercept, x_intercept_err, chi2ndf =  linear_fit(vols,gains,gain_errors)
        if len(gains) >= 2:
            slope, intercept, x_intercept , slope_err, intercept_err, x_intercept_err, chi2, ndf =  linear_fit_bootstrap(vols,gains, gain_errors, 500)
        else:
            slope, intercept, x_intercept , slope_err, intercept_err, x_intercept_err, chi2, ndf = 0.,0.,0.,0.,0.,0.,0.,0
        if ndf != 0:
            if chi2 / ndf > 0.1:
                slope, intercept, x_intercept , slope_err, intercept_err, x_intercept_err, chi2, ndf =  linear_fit_bootstrap(vols[1:],gains[1:], gain_errors[1:], 500)
        if len(rob_gains) >= 2:
            rob_slope, rob_intercept, rob_x_intercept , rob_slope_err, rob_intercept_err, rob_x_intercept_err, rob_chi2, rob_ndf =  linear_fit_bootstrap(rob_vols, rob_gains, rob_gain_errors, 500)
        else:
            rob_slope, rob_intercept, rob_x_intercept , rob_slope_err, rob_intercept_err, rob_x_intercept_err, rob_chi2, rob_ndf = 0.,0.,0.,0.,0.,0.,0.,0
        if rob_ndf != 0:
            if rob_chi2 / rob_ndf > 0.1:
                rob_slope, rob_intercept, rob_x_intercept , rob_slope_err, rob_intercept_err, rob_x_intercept_err, rob_chi2, rob_ndf =  linear_fit_bootstrap(rob_vols[1:], rob_gains[1:], rob_gain_errors[1:], 500)
        if ndf != 0:
            if chi2 / ndf > 0.000001:
                vols_plus = deepcopy(vols)
                vols_plus.insert(0, x_intercept)

                # Set the title of the plot
                plt.title(f"Linear regression of gain (run{run} po{position} ch{channel})")
                draw_linear_fit(fig, ax, vols,vols_plus ,gains, gain_errors, - x_intercept* slope, slope, x_intercept_err, chi2/ndf) 
                # Set the y-axis limits

                # Save the plot as a PDF file
                filename = f'lfit_run{run}_pos{position}_ch{channel}.pdf'
                
                plt.savefig(output_dir + "/pdf/" +filename)
                # Add the generated PDF file to the merger object
                #pdf_merger.append(output_dir + "/pdf/" +filename)
                # Clear the axes
                #plt.cla()
                # Clear the figure
                #fig.clf()
                plt.clf()
        for ov in range(1, 7):
            if len(gains) >= 2:
                gain = slope*( ov + init_vol - x_intercept) 
                df_dslope = ov + init_vol - x_intercept
                # Error propagation formula
                gain_err = math.sqrt((df_dslope * slope_err)**2 + (slope * x_intercept_err)**2)
            else:
                gain = 0.
                gain_err = 0.
            fitted_gain.append(gain)
            fitted_gain_err.append(gain_err)
            chi2_list.append(chi2)
            ndf_list.append(ndf)
            slope_list.append(slope)
            slope_err_list.append(slope_err)
            vbd_dict.append(x_intercept)
            ov_dict.append(init_vol + ov - x_intercept)
            vbd_err_dict.append(x_intercept_err)
            rob_vbd_dict.append(rob_x_intercept)
            rob_vbd_err_dict.append(rob_x_intercept_err)
        for i in range(6):
            voltage_missing_list_all.append(voltage_missing_list[i])
            fit_failure_list_all.append(fit_failure_list[i])
            large_chi2_list_all.append(large_chi2_list[i])
            parameter_miss_list_all.append(parameter_miss_list[i])

        
    df_tmp['vbd'] = vbd_dict
    df_tmp['ov'] = ov_dict
    df_tmp['vbd_err'] = vbd_err_dict
    df_tmp['fit_gain'] = fitted_gain
    df_tmp['fit_gain_err'] = fitted_gain_err
    df_tmp['prefit_gain'] = prefit_gain
    df_tmp['prefit_gain_err'] = prefit_gain_err
    df_tmp['slope'] = slope_list
    df_tmp['slope_err'] = slope_err_list
    df_tmp['linear_chi2'] = chi2_list
    df_tmp['ndf'] = ndf_list
    df_tmp['rob_vbd'] = rob_vbd_dict
    df_tmp['rob_vbd_err'] = rob_vbd_err_dict
    df_tmp['pos'] = [position for i in range(96)]
    df_tmp['run'] = [run for i in range(96)]
    df_tmp['ch'] = [ch for ch in range(1,17) for _ in range(6)]
    df_tmp['vol'] = [ov for _ in range(16) for ov in range(1, 7)]
    df_tmp['vol_missing'] = voltage_missing_list_all
    df_tmp['fit_failure'] = fit_failure_list_all
    df_tmp['large_chi2'] = large_chi2_list_all
    df_tmp['params_missing'] = parameter_miss_list_all
    #df_tmp.to_csv(f"{output_dir}/csv/vbd_tile{position}.csv", index=False)
      
    file_path = f"{output_dir}/csv/get_vbd_run{run}.csv"
    # Check if file exists and is empty
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        # File doesn't exist or is empty, so write DataFrame with header
        df_tmp.to_csv(file_path, index=False)
    else:
        # File exists and is not empty, so append without header
        df_tmp.to_csv(file_path, index=False)
