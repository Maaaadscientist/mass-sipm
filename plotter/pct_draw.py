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
maxcolor="red"
#maxcolor="crimson"
mincolor="deepskyblue"
#mincolor="dodgerblue"
labelsize=28
titlesize=40
textsize=24
voltages = ["3v", "3p5v", "4v", "4p5v", "vop"]
lower_edges = {"3v":0, "3p5v":0, "4v":0, "4p5v":0, "vop":0}
upper_edges = {"3v":0.4, "3p5v":0.4, "4v":0.4, "4p5v":0.4, "vop":0.4}
lower_edges_hist = {"3v":0.05, "3p5v":0.05, "4v":0.08, "4p5v":0.08, "vop":0.08}
upper_edges_hist = {"3v":0.24, "3p5v":0.24, "4v":0.25, "4p5v":0.28, "vop":0.25}
step = 0.01
pct_max = {v: np.zeros(n, dtype=float) for v in voltages}
pct_max_err = {v: np.zeros(n, dtype=float) for v in voltages}
pct_min = {v: np.zeros(n, dtype=float) for v in voltages}
pct_min_err = {v: np.zeros(n, dtype=float) for v in voltages}
pct_mean = {v: np.zeros(n, dtype=float) for v in voltages}
pct_mean_err = {v: np.zeros(n, dtype=float) for v in voltages}
tsn = np.zeros(n, dtype=float)
count = np.zeros(n, dtype=float)
for i, event in enumerate(tree):
    count[i] = i+1
    tsn[i] = event.tsn
    for v in voltages:
        pct_max[v][i] = abs(getattr(event, f"pct_{v}_max"))
        pct_max_err[v][i] = abs(getattr(event, f"pct_{v}_max_err"))
        pct_min[v][i] = abs(getattr(event, f"pct_{v}_min"))
        pct_min_err[v][i] = abs(getattr(event, f"pct_{v}_min_err"))
        pct_mean[v][i] = abs(getattr(event, f"pct_{v}_mean"))
        pct_mean_err[v][i] = abs(getattr(event, f"pct_{v}_mean_err"))
        #setattr(pct_max[v], i, abs(getattr(event, f"pct_{v}_max")))
        #setattr(pct_max_err[v], i, abs(getattr(event, f"pct_{v}_max_err")))
        #setattr(pct_min[v], i, abs(getattr(event, f"pct_{v}_min")))
        #setattr(pct_min_err[v], i, abs(getattr(event, f"pct_{v}_min_err")))
        #setattr(pct_mean[v], i, abs(getattr(event, f"pct_{v}_mean")))
        #setattr(pct_mean_err[v], i, abs(getattr(event, f"pct_{v}_mean_err")))
# Initialize a single figure for all plots
plt.figure(figsize=(32, 24))  # Adjust the figure size as needed
for idx,v in enumerate(voltages):
    #plt.figure(figsize=(30, 3))
    # Plot max values
    # Create a subplot
    plt.subplot(len(voltages), 1, idx + 1)

    plt.errorbar(count, pct_max[v], yerr=pct_max_err[v], fmt='^', label="Max", capsize=3, color=maxcolor, markersize=3, alpha=0.7, mfc='none', elinewidth=0.5, markeredgewidth=0.5)
    
    # Plot min values
    plt.errorbar(count, pct_min[v], yerr=pct_min_err[v], fmt='v', label="Min", capsize=3, color=mincolor, markersize=3, alpha=0.7, elinewidth=0.5, mfc='none', mec=meancolor, markeredgewidth=0.5)
    
    # Plot mean values
    plt.errorbar(count, pct_mean[v], yerr=pct_mean_err[v], fmt='o', label="Mean", capsize=3, color=meancolor,markersize=3, alpha=0.7, elinewidth=0.5, mfc='none', mec=meancolor, markeredgewidth=0.5)
    
    plt.axhline(y=0.15, color='lime', linestyle='--', linewidth=2.5, label="MAX 15\%", zorder=5)
    ov = v.replace("v","V").replace("p", ".") if v!="vop" else "$\\mathrm{V}_{op}$"
    if idx == len(voltages) - 1:  # Only label the x-axis on the bottom-most subplot
        plt.xlabel("SiPM Tile Number", fontsize=labelsize)
    if idx == 0:
        plt.title(f"Crosstalk Probabilities of different SiPM tiles", fontsize=titlesize)
    #plt.xlabel("SiPM Tile Number")
    plt.xticks(fontsize=labelsize)  # Adjust font size for x-axis ticks
    plt.yticks(fontsize=labelsize)  # Adjust font size for y-axis ticks
    plt.ylabel("Crosstalk Probability", fontsize=labelsize)
    plt.ylim(lower_edges[v],upper_edges[v])
    plt.xlim(-100, max(count) + 400)
    # Add a text label for the voltage
    xlim = plt.xlim()  # Get current x-axis limits
    ylim = plt.ylim()  # Get current y-axis limits
    x_pos = xlim[1] - 0.01 * (xlim[1] - xlim[0])  # 5% from the right edge
    y_pos = ylim[1] - 0.04 * (ylim[1] - ylim[0])  # 5% from the top edge
    plt.text(x_pos, y_pos, f"Over Voltage: {ov}", ha="right", va="top", fontsize=labelsize, color="black")  # Adjust font size as needed

    plt.grid(True)
    plt.legend(fontsize=labelsize,loc='center right')
    
plt.tight_layout()
plt.subplots_adjust(left=0.05)
plt.savefig("pct_tiles.pdf")

for v in voltages:
    ##file.Close()
    #bins = [0.4, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66,0.68, 0.7]
    bins = np.arange(lower_edges_hist[v], upper_edges_hist[v] + step, step)
    hist_max, _ = np.histogram(pct_max[v], bins=bins)
    hist_min, _ = np.histogram(pct_min[v], bins=bins)
    hist_mean, _ = np.histogram(pct_mean[v], bins=bins)
    
    # Determine the center of the bins to use for plotting
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    
    bin_width = np.min(np.diff(bins)) / 3  # Assuming we're plotting 3 histograms side by side
    plt.figure(figsize=(20, 15))
    
    # Determine the offset for each histogram to center them around bin center
    offset = bin_width / 2
    
    # Plot the histograms side by side
    bar_max = plt.bar(bin_centers + offset, hist_max, width=bin_width, alpha=0.7, label="Max", align='center', color=maxcolor)
    bar_min = plt.bar(bin_centers - offset, hist_min, width=bin_width, alpha=0.7, label="Min", align='center', color=mincolor)
    bar_mean = plt.bar(bin_centers, hist_mean, width=bin_width, alpha=0.7, label="Mean", align='center', color=meancolor)
    
    plt.axvline(x=0.15, color='lime', linestyle='--', linewidth=5, label="Req. $\\leq$ 0.15", zorder=5)
    # Add text on top of the bars
    def add_text_on_bars(bars, color):
        for bar in bars:
            height = bar.get_height()
            if height != 0:  # Only add text if height is not zero
                plt.text(bar.get_x() + bar.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom', fontsize=textsize, color=color)

    add_text_on_bars(bar_max, maxcolor)
    add_text_on_bars(bar_min, mincolor)
    add_text_on_bars(bar_mean, meancolor)
    xlim = plt.xlim()  # Get current x-axis limits
    ylim = plt.ylim()  # Get current y-axis limits
    x_pos = xlim[1] - 0.05 * (xlim[1] - xlim[0])  # 5% from the right edge
    y_pos = ylim[1] - 0.4 * (ylim[1] - ylim[0])  # 5% from the top edge
    plt.text(x_pos, y_pos, f"Total: {int(max(count))}", ha="right", va="top", fontsize=labelsize + 2, color="black")  # Adjust font size as needed
    # Set x-axis ticks to match the binning
    plt.xticks(bins, fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    
    ov = v.replace("v","V").replace("p", ".") if v!="vop" else "$\\mathrm{V}_{op}$"
    plt.xlabel("Crosstalk Probability",fontsize=titlesize)
    plt.ylabel("Number of Tiles",fontsize=titlesize)
    plt.legend(fontsize=titlesize)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title(f"Crosstalk Probability ({ov})", fontsize=titlesize+6)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    plt.savefig(f"pct_distribution_{v}.pdf")
    plt.clf()


