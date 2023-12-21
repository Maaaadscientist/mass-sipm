import os
import sys
import re
import ROOT
from ROOT import TFile, TH2F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.colors import Normalize

from matplotlib.colors import LogNorm
import scienceplots
plt.style.use('science')
plt.style.use('nature')

labelsize=28
titlesize=40
textsize=24
# Choose a base colormap
#base_cmap = plt.cm.magma
base_cmap = plt.cm.plasma

# Create a new colormap from the base one
# ListedColormap takes a list of colors or a colormap object
# You're taking all colors from the base colormap except the first one and adding white at the beginning
custom_cmap = mcolors.ListedColormap(['white'] + [base_cmap(i) for i in range(1, base_cmap.N)])

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

input_path = os.path.abspath(sys.argv[1])
csv_path = os.path.abspath(sys.argv[2])

file_info = fetch_file_info(input_path.split("/")[-1])
run = int(file_info['run'])
vol = int(file_info['ov'])
channel = int(file_info['channel'])

df = pd.read_csv(csv_path)
file = TFile(input_path, 'READ')
selected_channel = 5  # for example, channel 5
channel_data = df[(df['pos'] == selected_channel) & (df['run'] == run) & (df['vol'] == vol) & (df['ch'] == channel)]
hist_name = f"waveform_ch{selected_channel}"
hist = file.Get(hist_name)
if not hist:
    print(f"Histogram {hist_name} not found in the file.")
    exit()
# Get the number of bins in each dimension
y_min, y_max = 70, 100  # define your range here
z_min, z_max = 0, 2000  # define your Z range here
x_min, x_max = 1250, 1350  # define your range here

# Find bin numbers corresponding to the y range
y_bin_min = hist.GetYaxis().FindBin(y_min)
y_bin_max = hist.GetYaxis().FindBin(y_max)

# Find bin numbers corresponding to the x range
x_bin_min = hist.GetXaxis().FindBin(x_min)
x_bin_max = hist.GetXaxis().FindBin(x_max)

# Adjust the size of your NumPy array to match the new x range
hist_array = np.zeros((x_bin_max - x_bin_min + 1, y_bin_max - y_bin_min + 1))

# Fill the array with bin contents from the TH2F, considering only the specified x and y ranges
for i in range(x_bin_min, x_bin_max+1):
    for j in range(y_bin_min, y_bin_max+1):
        hist_array[i-x_bin_min][j-y_bin_min] = hist.GetBinContent(i, j)
# Define the normalization: since you want white for 0, set the 'vmin' to a small number above 0
#norm = mcolors.Normalize(vmin=0.0001, vmax=hist_array.max() - 1000)
#
#plt.imshow(hist_array.T, origin='lower', aspect='auto', 
#           extent=[x_min, x_max, y_min, y_max],
#           cmap=custom_cmap, norm=norm)
## Ensure no zero or negative values in the data for the log scale
##hist_array[hist_array <= 0] = np.min(hist_array[hist_array > 0])
##
### Create a custom colormap as before
##custom_cmap = mcolors.ListedColormap(['white'] + [plt.cm.viridis(i) for i in range(1, plt.cm.viridis.N)])
##
### Apply LogNorm for the color scaling
##plt.imshow(hist_array.T, origin='lower', aspect='auto',
##           extent=[hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax(), y_min, y_max],
##           cmap=custom_cmap, norm=LogNorm(vmin=hist_array.min(), vmax=hist_array.max()))
#
#plt.colorbar()
#plt.title(f"2D Heatmap of {hist_name} with Y-range [{y_min}, {y_max}]")
#plt.xlabel('X Axis Title')
#plt.ylabel('Y Axis Title')
#plt.show()

# Assuming there's only one row that matches, otherwise you might need to handle multiple rows
if not channel_data.empty:
    baseline_position = channel_data['bl'].iloc[0]  # .iloc[0] accesses the first row of the filtered data
    baseline_rms = channel_data['bl_rms'].iloc[0]
    mu =  channel_data['mu'].iloc[0]
    ref_mu =  channel_data['ref_mu'].iloc[0]
    ref_mu_err =  channel_data['ref_mu_err'].iloc[0]
    mu_err =  channel_data['mu_err'].iloc[0]
    lambda_ = channel_data['lambda'].iloc[0]
    lambda_err = channel_data['lambda_err'].iloc[0]
    gain = channel_data['gain'].iloc[0]
    n_peaks = channel_data['n_peaks'].iloc[0]
    events = channel_data['events'].iloc[0]
    dcr = channel_data['dcr'].iloc[0]
    dcr_err = channel_data['dcr_err'].iloc[0]
    
    batch = channel_data['batch'].iloc[0]
    box = channel_data['box'].iloc[0]
    tsn = channel_data['tsn'].iloc[0]

    vbd = channel_data['vbd'].iloc[0]
    vbd_err = channel_data['vbd_err'].iloc[0]

else:
    print(f"No data found for Channel {selected_channel}")

plt.figure(figsize=(20, 15))

norm = Normalize(vmin=z_min, vmax=z_max)

plt.imshow(hist_array.T, origin='lower', aspect='auto',
           extent=[x_min, x_max, y_min, y_max],  # Adjusted for selected x and y ranges
           cmap=custom_cmap, norm=norm)  # Use the linear normalization here
#plt.title(f"2D Heatmap of {hist_name} with X-range [{x_min}, {x_max}], Y-range [{y_min}, {y_max}], and Z-range [{z_min}, {z_max}]")
param_text = (f"SN: {batch}-{box}-{int(tsn)}-{channel}\n"
              "Events:"+f" {events}\n"
              "$\\mathrm{Amp}_\\mathrm{\\,baseline}$:"+f" {baseline_position:.3f} (mV)\n"
              "$\\mathrm{RMS}_\\mathrm{\\,baseline}$:"+f" {baseline_rms:.3f} (mV)\n"
              "Recognised Peaks :"+f" {n_peaks}\n"
              "\n"
              "$\\mathrm{V}_\\mathrm{bd}$ :"+f" {vbd:.2f}"+" $\\pm$ "+f"{vbd_err:.3f} (V)\n"
              "$\\mu$ :"+f" {mu:.3f}"+" $\\pm$ "+f"{mu_err:.3f}\n"
              "$\\mu_\\mathrm{ref.}$ :"+f" {ref_mu:.3f}"+" $\\pm$ "+f"{ref_mu_err:.3f}\n"
              "$\\lambda$ :"+f" {lambda_:.3f}"+" $\\pm$ "+f"{lambda_err:.3f}\n"
              "DCR:"+f" {dcr:.1f}"+" $\\pm$ "+f"{dcr_err:.1f}"+" ($\\mathrm{Hz}/\\mathrm{mm}^2$)\n"
              "Gain :"+f" {gain:.3f} (pC)")
plt.text(0.65, 0.85, param_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=textsize)

plt.tick_params(axis='x', which='major', pad=10)  # Increase padding for x-axis labels
plt.tick_params(axis='y', which='major', pad=15)  # Increase padding for y-axis labels

plt.xticks(fontsize=labelsize)
plt.yticks(fontsize=labelsize)
# Create the heatmap and colorbar
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=labelsize)  # Set the font size here

plt.title(f"Waveform Heat Map (LED Signal Range)", fontsize=titlesize, pad=25)
plt.xlabel('Time (8ns)', fontsize=labelsize, labelpad=15)
plt.ylabel('Amplitude (mV)', fontsize=labelsize, labelpad=15)
plt.savefig("waveform_SR.pdf")

file.Close()

