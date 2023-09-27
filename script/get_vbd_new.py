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

def remove_outliers(x_list, y_list, y_err_list, threshold=2):
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


def draw_linear_fit(fig, ax, x, x_with_bkv, y, y_err, intercept, slope, x_intercept_err, chi2):
    # convert input lists to numpy arrays
    x = np.array(x)
    xplus = np.array(x_with_bkv)
    y = np.array(y)
    y_err = np.array(y_err)

    # Define the fitted line
    fit = slope * xplus + intercept
    # Plot the data points with error bars
    ax.errorbar(x, y, yerr=y_err, fmt='o',markersize=2, color='black', label='Charge Data')

    # Plot the fitted line
    ax.plot(xplus, fit, color='red', label='Charge fit')
    # Set the limits for the twin y-axis
    ax.set_ylim([0, np.max(y) * 1.1])

    # Add labels and legend
    ax.set_xlabel('Voltage(V)')
    ax.set_ylabel(f'Charge')
    ax.legend()
    # Add chi-square value
    ax.text(0.05, 0.75, f'$\chi^2$/ndf: {chi2:.5f}', transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.65, f'Breakdown Voltage: {x_with_bkv[0]:.3f} $\pm$ {x_intercept_err:.3f}', transform=ax.transAxes, verticalalignment='top')

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
    # Calculate the sum of squared residuals (SSR)
    SSR = np.sum(squared_residuals)
    #print(SSR)
    degrees_of_freedom = len(x) - 2  # Number of data points minus number of fitted parameters
    reduced_chi2 = SSR / degrees_of_freedom

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
    x_intercept_err = np.sqrt(x_intercept_std_err ** 2 + (reduced_chi2 / slope) ** 2)

    return (
        slope,
        intercept,
        x_intercept,
        slope_std_err,
        intercept_std_err,
        x_intercept_err,
        reduced_chi2
    )


if len(sys.argv) < 3:
    print("Usage: python prepare_jobs <input_file> <output_dir>")
else:
   input_tmp = sys.argv[1]
   output_tmp = sys.argv[2]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
input_file =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
output_dir = os.path.abspath(output_tmp)
# Find a certain element based on other column values

if not os.path.isdir(output_dir + "/csv"):
    os.makedirs(output_dir + "/csv")

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


#vols = [i for i in range(2,7)]
for position in range(16):
    df_tmp = pd.DataFrame()
 
    vbd_dict = []
    vbd_rob_dict = []
    vbd_err_dict = []
    vbd_rob_err_dict = []
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
        for voltage in range(1,7):
            print(position, channel, voltage)
            filtered_df = df.loc[ (df['ch'] == channel) &
                            (df['pos'] == position) &(df['vol'] == voltage)]
            #chi2 = filtered_df.head(1)['chi2'].values[0]
            if len(filtered_df['rob_gain'].tolist()) != 0:
                rob_gain = filtered_df.head(1)['rob_gain'].values[0]
                rob_gain_err = filtered_df.head(1)['rob_gain_err'].values[0]
            if len(filtered_df['gain'].tolist()) != 0:
                gain = filtered_df.head(1)['gain'].values[0]
                gain_err = filtered_df.head(1)['gain_err'].values[0]
            if gain_err / gain < 0.05:
                gains.append(gain)
                gain_errors.append(gain_err)
                vols.append(voltage + 48)
            if rob_gain_err / rob_gain < 0.05:
                rob_gains.append(rob_gain)
                rob_gain_errors.append(rob_gain_err)
                rob_vols.append(voltage + 48)

        vols, gains, gain_errors = remove_outliers(vols, gains, gain_errors)
        rob_vols, rob_gains, rob_gain_errors = remove_outliers(rob_vols, rob_gains, rob_gain_errors)
        #slope, x_intercept, x_intercept_err, chi2ndf =  linear_fit(vols,gains,gain_errors)
        slope, intercept, x_intercept , slope_err, intercept_err, x_intercept_err, chi2ndf =  linear_fit_bootstrap(vols,gains, gain_errors, 500)
        rob_slope, rob_intercept, rob_x_intercept , rob_slope_err, rob_intercept_err, rob_x_intercept_err, rob_chi2ndf =  linear_fit_bootstrap(vols,gains, gain_errors, 500)
        vbd_dict.append(x_intercept)
        vbd_err_dict.append(x_intercept_err)
        rob_vbd_dict.append(rob_x_intercept)
        rob_vbd_err_dict.append(rob_x_intercept_err)
        
    df_tmp['vbd'] = vbd_dict
    df_tmp['vbd_err'] = vbd_err_dict
    df_tmp['rob_vbd'] = rob_vbd_dict
    df_tmp['rob_vbd_err'] = rob_vbd_err_dict
    df_tmp['position'] = [position for i in range(16)]
    df_tmp['channel'] = [ch for ch in range(1,17)]
    df_tmp.to_csv(f"{output_dir}/csv/vbd_tile{position}.csv", index=False)
      
