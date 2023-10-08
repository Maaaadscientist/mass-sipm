import os, sys
import pandas as pd
import numpy as np
from scipy import stats

import random
import matplotlib.pyplot as plt

invalid_runs = {"light":[130, 134, 187, 208, 229, 238, 242, 264, 265]}
def process_data(x_list, y_list, y_error_list):
    import numpy as np
    from collections import defaultdict
    
    # Identify indices where any of the values are nan
    valid_indices = [i for i, (x, y, y_err) in enumerate(zip(x_list, y_list, y_error_list))
                     if not (np.isnan(x) or np.isnan(y) or np.isnan(y_err))]
    
    # Filter the lists to exclude indices with nan values
    x_list = [x_list[i] for i in valid_indices]
    y_list = [y_list[i] for i in valid_indices]
    y_error_list = [y_error_list[i] for i in valid_indices]

    # Create a dictionary to store y and y_error values for each x value
    data_dict = defaultdict(list)
    
    for x, y, y_err in zip(x_list, y_list, y_error_list):
        data_dict[x].append((y, y_err))
    
    new_x_list = []
    new_y_list = []
    new_y_error_list = []
    
    # Process data
    for x, values in data_dict.items():
        if len(values) == 1:  # If only one y value for this x
            new_x_list.append(x)
            new_y_list.append(values[0][0])  # y value
            new_y_error_list.append(values[0][1])  # y_error value
        else:  # If multiple y values for this x
            y_values = [item[0] for item in values]
            y_errors = [item[1] for item in values]
            
            # Use mean for y value and std error for y error value
            mean_y = np.mean(y_values)
            std_error_y = np.sqrt(sum([e**2 for e in y_errors])) / len(y_errors)
            
            new_x_list.append(x)
            new_y_list.append(mean_y)
            new_y_error_list.append(std_error_y)
    
    return new_x_list, new_y_list, new_y_error_list
    
def linear_regression(x, y):
    n = len(x)
    
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum([i**2 for i in x])
    sum_y2 = sum([i**2 for i in y])
    sum_xy = sum([x[i]*y[i] for i in range(n)])
    
    m = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
    b = (sum_y - m*sum_x) / n
    
    # Residuals
    residuals = [y[i] - (m*x[i] + b) for i in range(n)]
    
    # Chi-squared
    chi2 = sum([r**2 for r in residuals])
    
    # Number of degrees of freedom
    ndf = n - 2
    
    # Pearson's correlation coefficient
    r = (n*sum_xy - sum_x*sum_y) / ((n*sum_x2 - sum_x**2) * (n*sum_y2 - sum_y**2))**0.5
    
    return m, b, chi2, ndf, r


# Reading the CSV file into a DataFrame
df = pd.read_csv('output.csv')

if not os.path.isdir("outputs"):
    os.mkdir("outputs")

new_x_grid = [0, 3.07, 9.07, 15.07, 21.07, 27.07, 33.07, 39.07, 45.07, 51.07, 57.07, 63.07]
new_y_grid = [0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0, -42.0, -48.0]

for remove_run_type, id_list in invalid_runs.items():
    for run_id in id_list:
        print("invalid", remove_run_type,"run", run_id, "has been removed")
        df = df[~((df['run_id'] == run_id) & (df['run_type'] == remove_run_type))]

csv_data = []

for x in new_x_grid:
    for y in new_y_grid:
        if x == 0 and y != 0:
            continue
        for index in range(1,17):
            # Filter the DataFrame based on the given position (x, y, index)
            filtered_df = df[(df['mu_index'] == index) & (df['new_x'] == x) & (df['new_y'] == y)]
            
            # Extract 'mu_value' and 'total_time' columns directly into lists
            mu_values = np.array(filtered_df['mu_value'].tolist())
            total_times = np.array(filtered_df['total_time'].tolist())
            mu_errors = np.array(filtered_df['mu_error'].tolist())
            
            if len(mu_values) == 0:
                print(x, y, index)
                continue
            # Calculate the mean and standard deviation of the mu_value
            mean_mu_value = sum(mu_values) / len(mu_values)
            std_mu_value = (sum([(mu - mean_mu_value) ** 2 for mu in mu_values]) / len(mu_values)) ** 0.5
            
            # Identify points that are 2-sigma away from the mean
            #outliers = filtered_df[(filtered_df['mu_value'] < mean_mu_value - 2 * std_mu_value) |
            #                       (filtered_df['mu_value'] > mean_mu_value + 2 * std_mu_value)]
            
            # Create a mask where True indicates an outlier
            is_outlier = filtered_df['mu_value'].apply(lambda mu: mu < mean_mu_value - 3 * std_mu_value or mu > mean_mu_value + 3 * std_mu_value)
            
            # Drop the outliers based on the mask
            filtered_df_no_outliers = filtered_df[~is_outlier]
            # Print the run_id and run_type of these points
            #for _, row in outliers.iterrows():
            #    print(f"Outlier found: run_id = {row['run_id']}, run_type = {row['run_type']}, mu_value = {row['mu_value']}")
            
            # Drop the outliers from filtered_df
            #iltered_df = filtered_df.drop(outliers.index)
            
            # The following processing now operates on filtered_df with the outliers removed
            mu_values = np.array(filtered_df['mu_value'].tolist())
            total_times = np.array(filtered_df['total_time'].tolist())
            mu_errors = np.array(filtered_df['mu_error'].tolist())

            # Filtered data without outliers
            mu_values_filtered = np.array(filtered_df_no_outliers['mu_value'].tolist())
            total_times_filtered = np.array(filtered_df_no_outliers['total_time'].tolist())
            mu_errors_filtered = np.array(filtered_df_no_outliers['mu_error'].tolist())
            
            
            time, mu, mu_error = process_data(total_times_filtered, mu_values_filtered, mu_errors_filtered)
            scaled_time = [t/1e9 for t in time]
            slope, intercept, chi2, ndf, r= linear_regression(time, mu)

            #plt.errorbar(, mu, yerr=mu_error, fmt='o', label='Data points', capsize=3, color='blue')
						# Plotting with the original data
            plt.errorbar(time, mu, yerr=mu_error, fmt='o', label='Data points', capsize=2, markersize=3, color='black')
            # Highlight the outliers
            outlier_df = filtered_df[is_outlier]
            plt.errorbar(outlier_df['total_time'], outlier_df['mu_value'], yerr=outlier_df['mu_error'], fmt='o', color='blue', label='Outliers', capsize=2, markersize=3)



            for _, row in outlier_df.iterrows():
                plt.text(row['total_time'], row['mu_value'], f"{row['run_id']}, {row['run_type']}", fontsize=8)
            
            # Annotating the plot with regression parameters
            params_text = f"Slope (m): {slope * 1e4:.2e}\nIntercept (b): {intercept:.2f}\n$\chi^2$: {chi2:.2e}\nndof: {ndf}"
            plt.annotate(params_text, xy=(0.03, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"))

            #print(f"Slope (m): {slope} +- {slope_err}")
            #print(f"Intercept (b): {intercept} +- {intercept_err}")
            y_fitted = [slope * xi + intercept for xi in time]
            plt.plot(time, y_fitted, 'r-', label='Fitted line')
            y_mean = sum(mu) / len(mu)  # Assuming 'mu' is your y-data
            ymin = y_mean - 0.05
            ymax = y_mean + 0.05
            plt.ylim(ymin, ymax)
            plt.xlim(0, 50000)
            # Adjusting x-axis tick labels
            locs, _ = plt.xticks()  # Get current tick locations and labels
            new_labels = [f"{loc/1e4:.2f}" for loc in locs]  # Compute new labels
            plt.xticks(locs, new_labels)  # Set new tick labels
            
            # Update the x-axis label to reflect the change in units
            plt.xlabel('Events (in 2.6e8)')
            plt.ylabel('Y-axis label')
            plt.legend(loc='lower right')
            plt.savefig(f"outputs/point_x{int(x+1e-8)}_y{int(y)}_index{index}.pdf")
            plt.clf()
            csv_data.append([x, y, index, slope, intercept, np.sqrt(chi2/ndf), ndf + 2])

# Convert the list to a DataFrame
csv_df = pd.DataFrame(csv_data, columns=['x', 'y', 'pos', 'slope', 'intercept', 'mu_unc', 'nPoints'])

# Export the DataFrame to a CSV file
csv_df.to_csv('mu_timeline.csv', index=False)
