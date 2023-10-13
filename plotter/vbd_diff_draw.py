import sys
import ROOT
import matplotlib.pyplot as plt
import numpy as np
file = ROOT.TFile(sys.argv[1], "READ")
tree = file.Get("tree")
n = tree.GetEntries()

plt.rcParams.update({
    "text.usetex": True,
    #"font.family": "serif",
    "font.family": "helvet",
    "text.latex.preamble": r"\usepackage{courier}",
})
meancolor="black"
maxcolor="darkorange"
#maxcolor="crimson"
mincolor="blue"
#mincolor="dodgerblue"
labelsize=28
titlesize=40
textsize=24
voltages = ["3v", "3p5v", "4v", "4p5v", "vop"]
lower_edge = 0 
upper_edge = 0.5 
lower_edge_hist = 0.01
upper_edge_hist = 0.39
step = 0.02
vbd_diff = np.zeros(n, dtype=float)
vbd_diff_err = np.zeros(n, dtype=float)
tsn = np.zeros(n, dtype=float)
count = np.zeros(n, dtype=float)
for i, event in enumerate(tree):
    count[i] = i+1
    tsn[i] = event.tsn
    vbd_diff[i] = abs(getattr(event, f"vbd_diff"))
    vbd_diff_err[i] = abs(getattr(event, f"vbd_diff_err"))
# Initialize a single figure for all plots
plt.figure(figsize=(32, 6))  # Adjust the figure size as needed
# Create a subplot

plt.errorbar(count, vbd_diff, yerr=vbd_diff_err, fmt='^', label="Max", capsize=4, color=meancolor, markersize=4, alpha=0.9, mfc='none', elinewidth=0.8, markeredgewidth=0.8)

plt.axhline(y=0.19, color='lime', linestyle='--', linewidth=2.5, label="Req. 0.19", zorder=5)
plt.xlabel("SiPM Tile Number", fontsize=labelsize)
plt.title(f"Breakdown Voltage Maximum Difference of SiPM tiles", fontsize=titlesize)
#plt.xlabel("SiPM Tile Number")
plt.xticks(fontsize=labelsize)  # Adjust font size for x-axis ticks
plt.yticks(fontsize=labelsize)  # Adjust font size for y-axis ticks
plt.ylabel("$\\mathrm{V}_\\mathrm{breakdown}$ difference (V)", fontsize=labelsize)
plt.ylim(lower_edge,upper_edge)
plt.xlim(-100, max(count) + 400)
# Add a text label for the voltage
xlim = plt.xlim()  # Get current x-axis limits
ylim = plt.ylim()  # Get current y-axis limits
x_pos = xlim[1] - 0.01 * (xlim[1] - xlim[0])  # 5% from the right edge
y_pos = ylim[1] - 0.04 * (ylim[1] - ylim[0])  # 5% from the top edge
#plt.text(x_pos, y_pos, f"Over Voltage: {ov}", ha="right", va="top", fontsize=labelsize, color="black")  # Adjust font size as needed

plt.grid(True)
plt.legend(fontsize=labelsize,loc='center right')
    
plt.tight_layout()
plt.subplots_adjust(left=0.05)
plt.savefig("vbd_diff_tiles.pdf")

##file.Close()
#bins = [0.4, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66,0.68, 0.7]
bins = np.arange(lower_edge_hist, upper_edge_hist + step, step)
hist_mean, _ = np.histogram(vbd_diff, bins=bins)
hist_neg, _ = np.histogram(vbd_diff - vbd_diff_err, bins=bins)
hist_pos, _ = np.histogram(vbd_diff + vbd_diff_err, bins=bins)

# Determine the center of the bins to use for plotting
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2

bin_width = np.min(np.diff(bins)) / 3  # Assuming we're plotting 3 histograms side by side
plt.figure(figsize=(20, 15))

# Determine the offset for each histogram to center them around bin center
offset = bin_width / 2

# Plot the histograms side by side
bar_pos = plt.bar(bin_centers + offset, hist_pos, width=bin_width, alpha=0.7, label="$\\mathbin{+\\phantom{+}}$1 $\\sigma$", align='center', color=maxcolor)
bar_neg = plt.bar(bin_centers - offset, hist_neg, width=bin_width, alpha=0.7, label="$\\mathbin{-\\phantom{+}}$1 $\\sigma$", align='center', color=mincolor)
bar_mean = plt.bar(bin_centers, hist_mean, width=bin_width, alpha=0.7, label="$\\mathrm{V}_\\mathrm{bd}$ diff.", align='center', color=meancolor)

plt.axvline(x=0.19, color='red', linestyle='--', linewidth=5, label="Req. $\\leq$ 0.19", zorder=5)
# Add text on top of the bars
def add_text_on_bars(bars, color):
    for bar in bars:
        height = bar.get_height()
        if height != 0:  # Only add text if height is not zero
            plt.text(bar.get_x() + bar.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom', fontsize=textsize, color=color)

xlim = plt.xlim()  # Get current x-axis limits
ylim = plt.ylim()  # Get current y-axis limits
x_pos = xlim[1] - 0.05 * (xlim[1] - xlim[0])  # 5% from the right edge
y_pos = ylim[1] - 0.4 * (ylim[1] - ylim[0])  # 5% from the top edge
plt.text(x_pos, y_pos, f"Total: {int(max(count))}", ha="right", va="top", fontsize=labelsize + 2, color="black")  # Adjust font size as needed
add_text_on_bars(bar_mean, meancolor)
add_text_on_bars(bar_neg, "dodgerblue")
add_text_on_bars(bar_pos, maxcolor)
# Set x-axis ticks to match the binning
plt.xticks(bins, fontsize=labelsize)
plt.yticks(fontsize=labelsize)

plt.xlabel("$\\mathrm{V}_\\mathrm{breakdown}$ difference (V)",fontsize=titlesize)
plt.ylabel("Number of Tiles",fontsize=titlesize)
plt.legend(fontsize=titlesize)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.title(f"Breakdown Voltage Maximum Difference", fontsize=titlesize+6)
#plt.tight_layout()
plt.subplots_adjust(left=0.08)

plt.savefig(f"vbd_diff_distribution.pdf")
plt.clf()


hist_mean, _ = np.histogram(vbd_diff, bins=bins)
hist_1sigma, _ = np.histogram(vbd_diff - vbd_diff_err, bins=bins)
hist_3sigma, _ = np.histogram(vbd_diff - 3 * vbd_diff_err, bins=bins)

# Determine the center of the bins to use for plotting
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2

bin_width = np.min(np.diff(bins)) / 3  # Assuming we're plotting 3 histograms side by side
plt.figure(figsize=(20, 15))

# Determine the offset for each histogram to center them around bin center
offset = bin_width / 2

# Plot the histograms side by side
bar_3sigma = plt.bar(bin_centers - offset, hist_3sigma, width=bin_width, alpha=0.7, label="$\\mathbin{-\\phantom{+}}$3 $\\sigma$", align='center', color="mediumorchid")
bar_1sigma = plt.bar(bin_centers, hist_1sigma, width=bin_width, alpha=0.7, label="$\\mathbin{-\\phantom{+}}$1 $\\sigma$", align='center', color=mincolor)
bar_mean = plt.bar(bin_centers + offset, hist_mean, width=bin_width, alpha=0.7, label="$\\mathrm{V}_\\mathrm{bd}$ diff.", align='center', color=meancolor)

plt.axvline(x=0.19, color='red', linestyle='--', linewidth=5, label="Req. $\\leq$ 0.19", zorder=5)
# Add text on top of the bars
def add_text_on_bars(bars, color):
    for bar in bars:
        height = bar.get_height()
        if height != 0:  # Only add text if height is not zero
            plt.text(bar.get_x() + bar.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom', fontsize=textsize, color=color)

xlim = plt.xlim()  # Get current x-axis limits
ylim = plt.ylim()  # Get current y-axis limits
x_pos = xlim[1] - 0.05 * (xlim[1] - xlim[0])  # 5% from the right edge
y_pos = ylim[1] - 0.4 * (ylim[1] - ylim[0])  # 5% from the top edge
plt.text(x_pos, y_pos, f"Total: {int(max(count))}", ha="right", va="top", fontsize=labelsize + 2, color="black")  # Adjust font size as needed
add_text_on_bars(bar_mean, meancolor)
add_text_on_bars(bar_1sigma, "dodgerblue")
add_text_on_bars(bar_3sigma, "mediumorchid")
# Set x-axis ticks to match the binning
plt.xticks(bins, fontsize=labelsize)
plt.yticks(fontsize=labelsize)

plt.xlabel("$\\mathrm{V}_\\mathrm{breakdown}$ difference (V)",fontsize=titlesize)
plt.ylabel("Number of Tiles",fontsize=titlesize)
plt.legend(fontsize=titlesize)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.title(f"Breakdown Voltage Maximum Difference", fontsize=titlesize+6)
#plt.tight_layout()
plt.subplots_adjust(left=0.08)

plt.savefig(f"vbd_diff_distribution_3sigma.pdf")
plt.clf()
