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
#from PyPDF2 import PdfMerger

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


#def linear_fit(x_list, y_list, y_err_list):
#    # convert input lists to numpy arrays
#    x = np.array(x_list)
#    y = np.array(y_list)
#
#    # perform linear regression
#    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#    
#    # calculate x-intercept
#    x_intercept = -intercept/slope
#    
#    return slope, intercept, x_intercept


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

def linear_fit(x_list, y_list, y_err_list):
    # Add a constant column to X
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    y_err_list = np.array(y_err_list)
    X = sm.add_constant(x_list)
    
    # Perform weighted linear regression
    #model = sm.WLS(y_list, X, weights=1 / y_err_list**2)  # Using WLS instead of OLS with weights
    model = sm.RLM(y_list, X, M=sm.robust.norms.HuberT()) 
    results = model.fit()
    
    # Extract slope, intercept, and their errors
    slope = results.params[1]
    intercept = results.params[0]
    slope_error = results.bse[1]
    intercept_error = results.bse[0]
    # Calculate x-intercept
    x_intercept = -intercept / slope
    
    # Calculate error of x-intercept
    x_intercept_err = np.sqrt((intercept_error / slope)**2 + (intercept * slope_error / slope**2)**2)
    # Calculate the predicted values
    y_pred = results.predict(X)
    
    # Calculate the residuals
    residuals = y_list - y_pred
    
    # Calculate the residual sum of squares (RSS)
    rss = np.sum(residuals**2)
    x_intercept_err = np.sqrt(x_intercept_err ** 2 + rss / slope ** 2)
    
    # Get the degrees of freedom
    degrees_of_freedom = len(X) - results.df_model - 1
    
    # Calculate the residual chi-square
    residual_chi2 = rss / results.scale

    return slope, x_intercept, x_intercept_err, residual_chi2

def linear_fitbkp(x_list, y_list, y_err_list):
    # Convert input lists to numpy arrays
    x = np.array(x_list)
    y = np.array(y_list)
    y_err = np.array(y_err_list)

    # Define the linear equation
    def linear_func(x, slope, m):
        return slope * x - m

    # Perform curve fitting with uncertainties
    #popt, pcov = curve_fit(linear_func, x, y, sigma=y_err, maxfev=5000)
    popt, pcov = curve_fit(linear_func, x, y, sigma=y_err, p0=[6, 47/6], maxfev=50000, method = 'lm')

    # Extract the slope and intercept from the fitted parameters
    slope, m = popt

    # Calculate the ±1 sigma confidence interval for the x_intercept
    slope_err, m_err = np.sqrt(np.diag(pcov))
    x_intercept = m / slope
    x_intercept_err = calculate_x_intercept_error(m, slope, m_err, slope_err)
    # Calculate the residuals
    residuals = y - linear_func(x, slope, m)

    # Calculate chi-square
    chi_square = np.sum((residuals / y_err) ** 2)
    # Calculate degrees of freedom
    dof = len(x) - len(popt)
    
    # Calculate reduced chi-square
    reduced_chi_square = chi_square / dof
    # Return the slope, intercept, and x_intercept with ±1 sigma confidence interval
    return slope, x_intercept, x_intercept_err, reduced_chi_square
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
    x_intercept_err = np.sqrt(x_intercept_std_err ** 2 + SSR / slope ** 2)

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
eos_mgm_url = "root://junoeos01.ihep.ac.cn"
directory = "/tmp/tao-sipmtest"
output_dir = os.path.abspath(output_tmp)
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


#vols = [i for i in range(2,7)]
for position in range(16):
  df_tmp = pd.DataFrame()
 
  vbd_dict = []
  vbd_err_dict = []
  # Create a PDF merger object
  #pdf_merger = PdfMerger()
  for channel in range(1,17):
    
    # Set up the plot
    fig, ax = plt.subplots()
    # Create a twin y-axis on the right side

    mean_diff_list = []
    mean_unc_list = []
    vols = []
    for voltage in range(1,7):
        print(position, channel, voltage)
        filtered_df = df.loc[ (df['ch'] == channel) &
                        (df['pos'] == position) &(df['vol'] == voltage)]
        #chi2 = filtered_df.head(1)['chi2'].values[0]
        # For ov = 1V, 2V, use rob_gain, for 5V, 6V ,use avg_gain
        if voltage == 1 or voltage == 2:
            if len(filtered_df['rob_gain'].tolist()) != 0:
                gain = filtered_df.head(1)['rob_gain'].values[0]
                gain_err = filtered_df.head(1)['rob_gain_err'].values[0]
            else:
                continue
        #elif voltage == 5 or voltage == 6:
        #    gain = filtered_df.head(1)['avg_gain'].values[0]
        #    gain_err = filtered_df.head(1)['gain_err'].values[0]
        else:
            if len(filtered_df['gain'].tolist()) != 0:
                gain = filtered_df.head(1)['gain'].values[0]
                gain_err = filtered_df.head(1)['gain_err'].values[0]
            else:
                continue

      #if chi2 < 5 and gain_err / gain < 0.02:
        mean_diff_list.append(gain)
        mean_unc_list.append(gain_err)
      # Clear the lists before each iteration
      #print("length of data:",len(mean_diff_list), "length of error:", len(mean_error_list))
        vols.append(voltage + 48)

    vols, mean_diff_list, mean_unc_list = remove_outliers(vols, mean_diff_list, mean_unc_list)
    #slope, x_intercept, x_intercept_err, chi2ndf =  linear_fit(vols,mean_diff_list,mean_unc_list)
    slope, intercept, x_intercept , slope_err, intercept_err, x_intercept_err, chi2ndf =  linear_fit_bootstrap(vols,mean_diff_list, mean_unc_list, 100)
    vols_plus = deepcopy(vols)
    vols_plus.insert(0, x_intercept)
    # Set the title of the plot
    ax.set_title(f"Linear regression (po{position} ch{channel})")
    draw_linear_fit(fig, ax, vols,vols_plus ,mean_diff_list, mean_unc_list, - x_intercept* slope, slope, x_intercept_err, chi2ndf) 
    vbd_dict.append(x_intercept)
    vbd_err_dict.append(x_intercept_err)
    # Set the y-axis limits

    # Save the plot as a PDF file
    filename = f'linear-regression_ch{channel}_sipm{position}.pdf'
    
    plt.savefig(output_dir + "/pdf/" +filename)
    # Add the generated PDF file to the merger object
    #pdf_merger.append(output_dir + "/pdf/" +filename)
    # Clear the axes
    ax.cla()
    # Clear the figure
    fig.clf()
    plt.clf()
  
  # Generate random values for the two arrays of breakdown voltage 
  size = 16  # Size of each array 
  breakdown_voltage = vbd_dict
  breakdown_voltage_err = vbd_err_dict 
  # Calculate mean and standard deviation
  mean_value = np.mean(breakdown_voltage)
  std_dev = np.std(breakdown_voltage)

  # Calculate the minimum and maximum values of breakdown_voltage
  min_value = min(breakdown_voltage)
  max_value = max(breakdown_voltage)
  
  # Set the y-range with a buffer
  buffer = 0.5  
  y_min = min_value - buffer 
  y_max = max_value + buffer 
  # Set the y-range
  plt.ylim(y_min, y_max)
   
  # Plotting the arrays 
  channels = np.arange(1, size + 1)  # X-axis values (channels) 
   
  #plt.plot(channels, breakdown_voltage, marker='o', label='Vbd (Amp)') 
  #plt.plot(channels, breakdown_voltage_2, marker='o', label='Vbd (Q)') 
  # Plot with error bars
  plt.errorbar(channels, breakdown_voltage, yerr=breakdown_voltage_err, marker='o', label='Vbd',capsize=5 , color = 'darkcyan')

  # Draw the mean line
  plt.axhline(y=mean_value, color='red', linestyle='--', label='Mean')
  # Draw the mean line
  plt.axhline(y=mean_value + 0.1, color='blue', linestyle='--', label='± 0.1V')
  plt.rcParams['axes.unicode_minus'] = False
  plt.axhline(y=mean_value - 0.1, color='blue', linestyle='--')

  
  x_min, x_max = plt.xlim()  # Get the limits of the x-axis
  # Draw the ±1 sigma band
  plt.fill_between(np.linspace(x_min, x_max, 1000),
                   mean_value - std_dev, mean_value + std_dev,
                   color='lightblue', alpha=0.3, label='±1 Sigma')
  
  # Draw the ±2 sigma band
  plt.fill_between(np.linspace(x_min, x_max, 1000),
                   mean_value - 2 * std_dev, mean_value + 2 * std_dev,
                   color='lightgreen', alpha=0.3, label='±2 Sigma')
   
  # Adding labels and title 
  plt.xlabel('Channel') 
  plt.ylabel('Breakdown Voltage') 
  plt.title(f'Breakdown Voltage Comparison (tile{position})') 
   
  # Adding legend 
  plt.legend() 
   
  # Customize the X-axis tick labels 
  plt.xticks(channels) 
   
  # Display the plot 
  plt.savefig(f'{output_dir}/pdf/vbd_allchannels_tile{position}.pdf') 
  plt.clf()
    
  df_tmp['vbd'] = vbd_dict
  df_tmp['vbd_err'] = vbd_err_dict
  df_tmp['position'] = [position for i in range(16)]
  df_tmp['channel'] = [ch for ch in range(1,17)]
  df_tmp.to_csv(f"{output_dir}/csv/vbd_tile{position}.csv", index=False)
# Close the figure to save memory
plt.close()
      
