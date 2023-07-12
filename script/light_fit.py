import os, sys
import re
import ROOT

if len(sys.argv) < 5:
    raise OSError("Usage: python light_fit.py <inputFile> <treeName> <branchName> <outputPath>")
else:
    input_tmp = sys.argv[1]
    tree_name = sys.argv[2]
    branch_name = sys.argv[3]
    output_tmp = sys.argv[4]
pattern = "(\w+)_ch(\d+)"
branch_match = re.match(pattern, branch_name)
if branch_match:
    print("matched")
    variable_name = branch_match.group(1)
    po = branch_match.group(2)

input_path = os.path.abspath(input_tmp)
f1 = ROOT.TFile(input_path)
tree = f1.Get(tree_name)
tree.Draw(f"baselineQ_ch{po}>>histogram")
histogram = ROOT.gPad.GetPrimitive("histogram") 
baseline = histogram.GetMean()
baseline_sigma = histogram.GetRMS()
hist = ROOT.TH1F("chargehist", "charge hist", 500, baseline - 3 * baseline_sigma, baseline + 3 * baseline_sigma)
tree.Draw(f"{branch_name}>>chargehist")
#hist = ROOT.gPad.GetPrimitive("chargehist")
sigQ = ROOT.RooRealVar("sigQ", "sigQ", baseline - 3 * baseline_sigma, baseline + 3 * baseline_sigma)
data = ROOT.RooDataHist("data","data", ROOT.RooArgSet(sigQ), hist)
mean = ROOT.RooRealVar("mean", "mean", baseline, baseline - baseline_sigma, baseline + baseline_sigma)
sigma = ROOT.RooRealVar("sigma", "sigma", 0.5,  0.1, baseline_sigma * 2)
gauss = ROOT.RooGaussian( "gauss", "gauss", sigQ, mean, sigma)
minimizer = ROOT.RooMinimizer(gauss.createNLL(data))
minimizer.setMinimizerType("Minuit2")  # Choose Minuit2 as the optimizer

# Perform the fit
result = minimizer.fit("")

total_entries = data.sumEntries()
# Create a normalization object for the Gaussian PDF
gauss_norm = gauss.createIntegral(ROOT.RooArgSet(sigQ), ROOT.RooFit.NormSet(sigQ), ROOT.RooFit.Range("sigQ"))
# Calculate the total number of events predicted by the signal PDF
total_events_signal = gauss_norm.getVal()  * total_entries
canvas = ROOT.TCanvas("c1","c1", 1200, 800)
# Divide the canvas into two asymmetric pads
pad1 =ROOT.TPad("pad1","This is pad1",0.05,0.05,0.72,0.97);
pad2 = ROOT.TPad("pad2","This is pad2",0.72,0.05,0.98,0.97);
pad1.Draw()
pad2.Draw()
pad1.cd()
frame = sigQ.frame()
frame.SetXTitle("Charge")
frame.SetYTitle("Events")
data.plotOn(frame)
#theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
#theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.Components("background"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
gauss.plotOn(frame)
frame.Draw()
# Create TLine objects for the legend
pad2.cd()
# Create a TPaveText to display parameter values and uncertainties
param_box = ROOT.TPaveText(0.01, 0.9, 0.9, 0.1, "NDC")
param_box.SetFillColor(ROOT.kWhite)
param_box.SetBorderSize(1)
param_box.SetTextFont(42)
param_box.SetTextSize(0.08)
param_box.AddText(f"#mean = {mean.getVal():.3f} #pm {mean.getError():.3f}")
param_box.AddText(f"#sigma = {sigma.getVal():.3f} #pm {sigma.getError():.3f}")
param_box.Draw("same")
canvas.SaveAs("last.pdf")

print(baseline_sigma)
print(tree.GetEntries())
print(total_events_signal)
print(hist.Integral())
mu = -ROOT.TMath.Log(hist.Integral()/tree.GetEntries())
print(mu)
