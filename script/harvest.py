import pandas as pd
import math
import re
import os, sys
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from copy import deepcopy
#from PyPDF2 import PdfMerger

import yaml
import ROOT

import statsmodels.api as sm
ref_map = {1 :[1 , 2 , 15 , 16], 2 :[3 ,  4, 14, 13], 3 :[4 , 5 , 12, 11], 4 :[7 , 8 , 10, 9 ],
           5 :[17, 18 ,31 , 32], 6 :[19, 20, 30, 29], 7 :[20, 21, 28, 27], 8 :[23, 24, 26, 25],
           9 :[33, 34 ,47 , 48], 10:[35, 36, 46, 45], 11:[36, 37, 44, 43], 12:[39, 40, 42, 41],
           13:[49, 50 ,63 , 64], 14:[51, 52, 62, 61], 15:[52, 53, 60, 59], 16:[55, 56, 58, 57],}
def get_vbd_difference(vbd_list, vbd_err_list):

    # Step 1: Find the maximum and minimum voltage values
    max_voltage = np.max(vbd_list)
    min_voltage = np.min(vbd_list)
    
    # Step 2: Calculate the difference between the maximum and minimum voltage values
    voltage_difference = max_voltage - min_voltage
    
    # Step 3: Find the corresponding errors for the maximum and minimum voltage values
    max_voltage_index = np.argmax(vbd_list)
    min_voltage_index = np.argmin(vbd_list)
    max_voltage_error = vbd_err_list[max_voltage_index]
    min_voltage_error = vbd_err_list[min_voltage_index]
    
    # Step 4: Calculate the combined error
    combined_error = math.sqrt(max_voltage_error**2 + min_voltage_error**2)
    # Step 5: Calculate the maximum voltage value minus the minimum voltage value and its corresponding error
    return voltage_difference, combined_error

def get_data_frame(input_path):
    if not os.path.isdir(input_path):
        df = pd.read_csv(input_path)
    else:
        all_data = []
        for filename in os.listdir(input_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(input_path, filename)
                data = pd.read_csv(file_path)
                all_data.append(data)
        df = pd.concat(all_data, ignore_index=True)
    return df
def filter_data_frame(df, po, ch, ov):
    filtered_df = df.loc[ (df['channel'] == ch) &
                        (df['position'] == po) & (df['voltage'] == ov) ]
    return filtered_df

def get_reff_mu(df_light, po, ch):
    mu = 0.
    mu_err_square = 0.
    for point in ref_map[ch]:
        filtered_df_light = df_light.loc[(df_light['position'] == po) & (df_light['point'] == point)]
        mu_point = filtered_df_light.head(1)['mu'].values[0]
        mu_err_point = filtered_df_light.head(1)['mu_err'].values[0]
        mu += mu_point
        mu_err_square += mu_err_point ** 2
    return mu , math.sqrt(mu_err_square) 

if len(sys.argv) < 4:
    print("Usage: python prepare_jobs <mother_dir> <run_info> <output_dir>")
else:
   input_tmp = sys.argv[1]
   run_info = sys.argv[2]
   output_tmp = sys.argv[3]
#file_list = "main_run_0075.txt"  # Path to the file containing the list of files
input_dir =  os.path.abspath(input_tmp)  # Path to the file containing the list of files
output_dir = os.path.abspath(output_tmp)
# Find a certain element based on other column values

pattern = "(\w+)_run_(\d+)"
matched = re.match(pattern, run_info)
if matched:
    run_type = matched.group(1)
    run_number = int(matched.group(2))
if not os.path.isdir(output_dir + "/csv"):
    os.makedirs(output_dir + "/csv")
if not os.path.isdir(output_dir + "/pdf"):
    os.makedirs(output_dir + "/pdf")

pdf_dir = output_dir + "/pdf/"
# Specify the path to your YAML file
yaml_path = os.path.abspath("valid-run.yaml")
with open(yaml_path, "r") as afile:
    yaml_data = yaml.safe_load(afile)
    if isinstance(yaml_data, dict):
        light_run_number = '{:04d}'.format(yaml_data.get(run_number))
    else:
        print("Invalid YAML data")
reff_pde_path = os.path.abspath("reff-pde.yaml")
with open(reff_pde_path, "r") as afile:
    pde_data = yaml.safe_load(afile)

# Access and manipulate the YAML data

vbd_dir = input_dir + "/vbd/" + run_info + "/csv"
signal_dir = input_dir + "/signal-fit/" + run_info + "/csv"
light_dir = input_dir +"/light-fit/" + f"light_run_{light_run_number}" + "/csv"
dcr_dir = input_dir + "/dcr-fit/" + run_info + "/csv"
# Read the CSV file into a pandas DataFrame
df_vbd = get_data_frame(vbd_dir)
df_dcr = get_data_frame(dcr_dir)
df_light = get_data_frame(light_dir)
df_signal = get_data_frame(signal_dir)

best_dict = {}
ov_dict = {}
ov_err_dict = {}
for key in ['dcr', 'pde','pct','vop','vbd', 'vbd_err']:
    best_dict[key] = np.zeros((16, 16))

for key in ['dcr', 'pde','pct', 'eps', 'vov']:
    ov_dict[key] = np.zeros((16, 16, 6))
    ov_err_dict[key] = np.zeros((16, 16, 6))

for ch in range(1, 17):
    dcr_tile = []
    vop_tile = []
    vbd_tile = []
    vbd_err_tile = []
    pde_tile = []
    pct_tile = []
    for po in range(16):
        #if po != 0 or ch != 1:
        #    continue
        
        filtered_df_vbd = df_vbd.loc[ (df_vbd['channel'] == ch) &
                        (df_vbd['position'] == po) ]
        vbd = filtered_df_vbd.head(1)['vbd'].values[0]
        vbd_err = filtered_df_vbd.head(1)['vbd_err'].values[0]
        #gain = filtered_df.head(1)['gain'].values[0]
        ov_list = []
        ov_err_list = []
        pde_list = []
        pde_err_list = []
        pct_list = []
        pct_err_list = []
        dcr_list = []
        delta_eps_list = []
        delta_eps_err_list = []
       
        
        mu_reff, mu_reff_err = get_reff_mu(df_light, po, ch)
        pde_reff = pde_data.get(po) 
        pde_reff_err = pde_reff * 0.1
        #dcr_err_list = []
        for ov in range(1,7): 
            #filtered_df_dcr = df_dcr.loc[ (df_dcr['channel'] == ch) &
             #           (df_dcr['position'] == po) & (df_dcr['voltage'] == ov) ]
            filtered_df_dcr = filter_data_frame(df_dcr, po, ch, ov)
            filtered_df_signal = filter_data_frame(df_signal, po, ch, ov)
            
            dcr = filtered_df_dcr.head(1)['dcr'].values[0]
            mu = filtered_df_signal.head(1)['mu'].values[0]
            mu_err = filtered_df_signal.head(1)['mu_err'].values[0]
            lambda_val = filtered_df_signal.head(1)['lambda'].values[0]
            lambda_err = filtered_df_signal.head(1)['lambda_err'].values[0]
            # Calculate Pct
            Pct = 1 - math.exp(-lambda_val)
            
            # Calculate Pct_err using error propagation formula
            Pct_err = abs(Pct * lambda_err * math.exp(-lambda_val))


            # Calculate pde_abs
            pde_abs = mu * pde_reff / mu_reff
            
            # Calculate pde_abs_err using error propagation formulas
            pde_abs_err = pde_abs * math.sqrt((mu_err / mu)**2 + (pde_reff_err / pde_reff)**2 + (mu_reff_err / mu_reff)**2)
            
            Pcn = Pct + 0.01 # currently AP is set to be 1%
            pde_eff = 0.51 + 0.35 * Pcn + 0.84 * Pcn**2 + (4.2*1e-4 + 2*1e-4* Pcn) * dcr
            delta_eps = pde_abs * 0.896 /0.9 - pde_eff

            ov_list.append(ov + 48 - vbd)
            ov_err_list.append(vbd_err)
            pde_list.append(pde_abs)
            pde_err_list.append(pde_abs_err)
            pct_list.append(Pct)
            pct_err_list.append(Pct_err)
            dcr_list.append(dcr)
            delta_eps_list.append(delta_eps)
            delta_eps_err_list.append(0.) # temporarily zero 
            #print(po, ch, ov ,dcr,  pde_abs, pde_abs_err, Pct, Pct_err, delta_eps)
        # perform spline interpolation
        ov_dict['eps'][ch - 1, po, :] = delta_eps_list
        ov_dict['pde'][ch - 1, po, :] = pde_list
        ov_dict['pct'][ch - 1, po, :] = pct_list
        ov_dict['dcr'][ch - 1, po, :] = dcr_list
        ov_dict['vov'][ch - 1, po, :] = ov_list
        ov_err_dict['vov'][ch - 1, po, :] = ov_err_list
        ov_err_dict['eps'][ch - 1, po, :] = delta_eps_err_list
        ov_err_dict['pde'][ch - 1, po, :] = pde_err_list
        ov_err_dict['pct'][ch - 1, po, :] = pct_err_list
        #ov_err_dict['dcr'][ch - 1, po, :] = dcr_err_list
        spl_eps = UnivariateSpline(ov_list, delta_eps_list)
        spl_dcr = UnivariateSpline(ov_list, dcr_list)
        spl_pde = UnivariateSpline(ov_list, pde_list)
        spl_pct = UnivariateSpline(ov_list, pct_list)
        # get interpolated y values
        xnew = np.linspace(0, 10, 1000)  # get more x values for smooth curve
        ynew = spl_eps(xnew)  # get interpolated y values
        
        # find the x at the maximum value of y
        idx = np.argmax(ynew)  # get the index of max y
        xmax = xnew[idx]  # get the corresponding x value
        # Check if xmax is within the range [2, 5]
        if xmax < 2 or xmax > 5:
            # If not, return the x value corresponding to the maximum y in the original data
            idx = np.argmax(delta_eps_list)  # get the index of max y in the original data
            xmax = ov_list[idx]  # get the corresponding x value
        print(xmax, spl_dcr(xmax))
        dcr_tile.append(spl_dcr(xmax))
        pde_tile.append(spl_pde(xmax))
        pct_tile.append(spl_pct(xmax))
        vop_tile.append(xmax)
        vbd_tile.append(vbd)
        vbd_err_tile.append(vbd_err)
        #plt.errorbar(ov_list, delta_eps_list, xerr=ov_err_list, yerr=delta_eps_err_list, fmt='-o', capsize=3)
        #
        #plt.xlabel('over voltage (V)')
        #plt.ylabel('$\Delta\epsilon$')
        #plt.xlim(0, 8)
        #ticks = np.arange(0, 8.1, 0.5)
        #plt.xticks(ticks)
        #plt.title(f'$\Delta\epsilon$ (run{run_number} tile{po} ch{ch})')
        #plt.grid(True)
        #plt.savefig(output_dir + "/pdf/"+ f"delta_eps_run{run_number}_tile{po}_ch{ch}.pdf")
    best_dict['dcr'][ch - 1, :] = dcr_tile
    best_dict['pde'][ch - 1, :] = pde_tile
    best_dict['pct'][ch - 1, :] = pct_tile
    best_dict['vop'][ch - 1, :] = vop_tile
    best_dict['vbd'][ch - 1, :] = vbd_tile
    best_dict['vbd_err'][ch - 1, :] = vbd_err_tile

vbd_diff_list = []
vbd_diff_err_list =[]
    
for po in range(16):
    vbd_diff, vbd_diff_err = get_vbd_difference(best_dict['vbd'][:,po],best_dict['vbd_err'][:, po])
    vbd_diff_list.append(vbd_diff)
    vbd_diff_err_list.append(vbd_diff_err)

x = np.arange(16)
################################### Vbd_diff / Tiles #####################################
fig0, ax0 = plt.subplots()
# Plot error bars with the same label and color
ax0.errorbar(x, vbd_diff_list,
                     yerr=vbd_diff_err_list, fmt='o', capsize=3, markersize=2, label="$V_{bd}$ Difference")

# Set labels and title
ax0.set_xlabel('Tiles')
ax0.set_ylabel('Vmax - Vmin')
ax0.set_title(f'Max breakdown voltage difference (Run {run_number})')
ax0.set_ylim(0, 0.5)

# Add a horizontal line at y = 0.5
ax0.axhline(y=0.19, color='red', linestyle='--',linewidth=2, label = 'Max Diff 0.19')
# Set x-axis ticks and labels
ax0.set_xticks(x)
ax0.set_xticklabels([str(i+1) for i in range(16)])

# Add a legend
ax0.legend()
      
plt.savefig(pdf_dir + f"vbd_diff_tiles_run{run_number}.pdf")
plt.clf()
################################### DCR / Tiles #####################################
fig1, ax1 = plt.subplots()
for ch in range(16):
    ax1.scatter(x, best_dict['dcr'][ch], label=f'Channel {ch+1}', s=5)

# Set labels and title
ax1.set_xlabel('Tiles')
ax1.set_ylabel('DCR')
ax1.set_title(f'Dark Counting Rates (Run {run_number})')

# Add a horizontal line at y = 0.5
ax1.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
ax1.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
# Set x-axis ticks and labels
ax1.set_xticks(x)
ax1.set_xticklabels([str(i+1) for i in range(16)])

# Add a legend
#ax.legend()
legend1 = ax1.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards


# Display the plot
plt.savefig(pdf_dir + f"dcr_tiles_run{run_number}.pdf")
plt.clf()

################################### Vbd / Tiles #####################################
fig2, ax2 = plt.subplots()
for ch in range(16):
    ax2.scatter(x, best_dict['vbd'][ch], label=f'Channel {ch+1}', s=5)
    
      
# Set labels and title
ax2.set_xlabel('Tiles')
ax2.set_ylabel('Vbd')
ax2.set_title(f'Breakdown Voltages (Run {run_number})')

# Add a horizontal line at y = 0.5
#ax2.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
#ax2.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
# Set x-axis ticks and labels
ax2.set_xticks(x)
ax2.set_xticklabels([str(i+1) for i in range(16)])

# Set grid lines on the y-axis
ax2.grid(axis='y')
# Set the y-axis limits
ax2.set_ylim(45, 48)
# Add a legend
#ax.legend()
legend2 = ax2.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards


# Display the plot
#plt.show()
plt.savefig(pdf_dir + f"vbd_tiles_run{run_number}.pdf")
plt.clf()

################################### PDE / Tiles #####################################

fig3, ax3 = plt.subplots()
for ch in range(16):
    ax3.scatter(x, best_dict['pde'][ch], label=f'Channel {ch+1}', s=5)
    
      
# Set labels and title
ax3.set_xlabel('Tiles')
ax3.set_ylabel('PDE')
ax3.set_title(f'Photon Detection Efficiencies (Run {run_number})')

# Add a horizontal line at y = 0.5
#ax2.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
#ax2.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
# Set x-axis ticks and labels
ax3.set_xticks(x)
ax3.set_xticklabels([str(i+1) for i in range(16)])

# Set grid lines on the y-axis
ax3.grid(axis='y')
# Set the y-axis limits
ax3.set_ylim(0., 1)
# Add a legend
#ax.legend()
legend3 = ax3.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards
plt.savefig(pdf_dir + f"pde_tiles_run{run_number}.pdf")
plt.clf()

################################### Pct / Tiles #####################################
fig4, ax4 = plt.subplots()
for ch in range(16):
    ax4.scatter(x, best_dict['pct'][ch], label=f'Channel {ch+1}', s=5)
    
      
# Set labels and title
ax4.set_xlabel('Tiles')
ax4.set_ylabel('Pct')
ax4.set_title(f'Crosstalk probability (Run {run_number})')

# Add a horizontal line at y = 0.5
#ax2.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
#ax2.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
# Set x-axis ticks and labels
ax4.set_xticks(x)
ax4.set_xticklabels([str(i+1) for i in range(16)])

# Set grid lines on the y-axis
ax4.grid(axis='y')
# Set the y-axis limits
ax4.set_ylim(0., 0.5)
# Add a legend
#ax.legend()
legend4 = ax4.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards
plt.savefig(pdf_dir + f"pct_tiles_run{run_number}.pdf")
plt.clf()

################################### Vop / Tiles #####################################
fig5, ax5 = plt.subplots()
for ch in range(16):
    ax5.scatter(x, best_dict['vop'][ch], label=f'Channel {ch+1}', s=5)
    
      
# Set labels and title
ax5.set_xlabel('Tiles')
ax5.set_ylabel('Vop')
ax5.set_title(f'Optimized Operating Over Voltage (Run {run_number})')

# Add a horizontal line at y = 0.5
#ax2.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
#ax2.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
# Set x-axis ticks and labels
ax5.set_xticks(x)
ax5.set_xticklabels([str(i+1) for i in range(16)])

# Set grid lines on the y-axis
ax5.grid(axis='y')
# Set the y-axis limits
ax5.set_ylim(2., 6)
# Add a legend
#ax.legend()
legend5 = ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards
plt.savefig(pdf_dir + f"vop_tiles_run{run_number}.pdf")
plt.clf()

################################### epsilon / ovs #####################################
# Create a dictionary to store combined labels
for ch in range(16):
    for po in range(16):
        # Generate unique label for each 'po' value
        label = f'tile {po}'

        # Assign a color based on 'po' value
        color = plt.cm.tab20(po % 20)  # Use modulo to cycle through colors

        # Plot error bars with the same label and color
        plt.errorbar(ov_dict['vov'][ch, po, :], ov_dict['eps'][ch, po, :], xerr=ov_err_dict['vov'][ch, po, :],
                     yerr=ov_err_dict['eps'][ch, po, :], fmt='-o', capsize=2, markersize=2, linewidth=1, label=label, color=color)


plt.xlabel('over voltage (V)')
plt.ylabel('$\Delta\epsilon$')
plt.xlim(0, 10)
ticks = np.arange(0, 10.1, 1)
plt.xticks(ticks)
plt.title(f'$\Delta\epsilon$ (Run {run_number})')
plt.grid(True)
# Update the legend with combined labels
handles, labels = plt.gca().get_legend_handles_labels()
combined_labels = [f"tile {po}" for po in range(16)]
plt.legend(handles, combined_labels, loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards
plt.savefig(pdf_dir + f"delta_epsilon_ovs_run{run_number}.pdf")
plt.clf()

################################### pde / ovs #####################################
        
# Create a dictionary to store combined labels
for ch in range(16):
    for po in range(16):
        # Generate unique label for each 'po' value
        label = f'tile {po}'

        # Assign a color based on 'po' value
        color = plt.cm.tab20(po % 20)  # Use modulo to cycle through colors

        # Plot error bars with the same label and color
        plt.errorbar(ov_dict['vov'][ch, po, :], ov_dict['pde'][ch, po, :], xerr=ov_err_dict['vov'][ch, po, :],
                     yerr=ov_err_dict['pde'][ch, po, :], fmt='-o', capsize=1, markersize=2, linewidth=1 , label=label, color=color)


plt.xlabel('over voltage (V)')
plt.ylabel('PDE')
plt.xlim(0, 10)
ticks = np.arange(0, 10.1, 1)
plt.xticks(ticks)
plt.title(f'Photon Detection efficiency (Run {run_number})')
plt.grid(True)
# Update the legend with combined labels
handles, labels = plt.gca().get_legend_handles_labels()
combined_labels = [f"tile {po}" for po in range(16)]
plt.legend(handles, combined_labels, loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards

plt.savefig(pdf_dir + f"pde_ovs_run{run_number}.pdf")
plt.clf()

################################### pct / ovs #####################################
# Create a dictionary to store combined labels
for ch in range(16):
    for po in range(16):
        # Generate unique label for each 'po' value
        label = f'tile {po}'

        # Assign a color based on 'po' value
        color = plt.cm.tab20(po % 20)  # Use modulo to cycle through colors

        # Plot error bars with the same label and color
        plt.errorbar(ov_dict['vov'][ch, po, :], ov_dict['pct'][ch, po, :], xerr=ov_err_dict['vov'][ch, po, :],
                     yerr=ov_err_dict['pct'][ch, po, :], fmt='-o', capsize=1, markersize=2, linewidth=1 , label=label, color=color)


plt.xlabel('over voltage (V)')
plt.ylabel('Pct')
plt.xlim(0, 10)
ticks = np.arange(0, 10.1, 1)
plt.xticks(ticks)
plt.title(f'Prompt Crosstalk Probability (Run {run_number})')
plt.grid(True)
# Update the legend with combined labels
handles, labels = plt.gca().get_legend_handles_labels()
combined_labels = [f"tile {po}" for po in range(16)]
plt.legend(handles, combined_labels, loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards

plt.savefig(pdf_dir + f"pct_ovs_run{run_number}.pdf")
plt.clf()

################################### dcr / ovs #####################################

# Create a dictionary to store combined labels
for ch in range(16):
    for po in range(16):
        # Generate unique label for each 'po' value
        label = f'tile {po}'

        # Assign a color based on 'po' value
        color = plt.cm.tab20(po % 20)  # Use modulo to cycle through colors

        # Plot error bars with the same label and color
        plt.errorbar(ov_dict['vov'][ch, po, :], ov_dict['dcr'][ch, po, :], xerr=ov_err_dict['vov'][ch, po, :],
                     yerr=ov_err_dict['dcr'][ch, po, :], fmt='-o', capsize=1, markersize=2, linewidth=1 , label=label, color=color)


plt.xlabel('over voltage (V)')
plt.ylabel('DCR')
plt.xlim(0, 10)
ticks = np.arange(0, 10.1, 1)
plt.xticks(ticks)
plt.title(f'Dark Counting Rate (Run {run_number})')
plt.grid(True)
# Add a horizontal line at y = 0.5
plt.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
plt.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
# Update the legend with combined labels
handles, labels = plt.gca().get_legend_handles_labels()
combined_labels = [f"tile {po}" for po in range(16)]
plt.legend(handles, combined_labels, loc='upper right', bbox_to_anchor=(1.4, 1.0))

# Adjusting the plot to accommodate the legend
plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards

plt.savefig(pdf_dir + f"dcr_ovs_run{run_number}.pdf")
plt.clf()

