import os,sys
import ROOT
from copy import deepcopy

file_name = "test.root"
th2f_name = "dir_run64_ov2_suffix/dcr2D_afterfilter0"
if len(sys.argv) == 2:
    file_name = sys.argv[1]
if len(sys.argv) == 3:
    file_name = sys.argv[1]
    th2f_name = sys.argv[2]
# Open the ROOT file containing the TH2F histogram
root_file = ROOT.TFile(file_name, "READ")

# Access the TH2F histogram
th2f_hist = root_file.Get("dir_run64_ov2_suffix/dcr2D_afterfilter0")

# Get the number of bins along the X-axis
num_bins_x = th2f_hist.GetNbinsX()

# Create a list to store the projected TH1F histograms
th1f_hists = []

# Loop over each bin along the X-axis
for bin_x in range(1, num_bins_x + 1):
    # Project the TH2F histogram onto the X-axis for the current bin
    th1f_proj = deepcopy(th2f_hist.ProjectionY("_py", bin_x, bin_x))
    
    # Append the projected TH1F histogram to the list
    th1f_hists.append(th1f_proj)

# Close the ROOT file
root_file.Close()

# Access the projected TH1F histograms in the list
# N bins == 200 for threshold from 0 to 4
# interval = 0.1, that is 40 bins
for i, th1f_proj in enumerate(th1f_hists):
    if not (i % 5 == 0):
        continue
    threshold = 4.0 / 200 * i
    # Do whatever you want with the projected TH1F histograms
    # For example, you can draw the histograms
    c = ROOT.TCanvas(f"c{i+1}", f"Canvas bin {i+1}")
    th1f_proj.Draw()
    c.SaveAs(r"dcrTimeAbove%.1f.pdf" % threshold)

