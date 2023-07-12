import os, sys
import re
import pandas as pd
import ROOT

ov_ranges = {1:100, 2:150, 3:200, 4:280, 5:360, 6: 420}
if len(sys.argv) < 6:
   print("Usage: python dcr_fit.py <input_file> <tree_name> <variable_name> <csv_path> <output_path> ")
else:
   input_tmp = sys.argv[1]
   tree_name = sys.argv[2]
   variable_name = sys.argv[3]
   csv_path = os.path.abspath(sys.argv[4])
   output_path = sys.argv[5]
# Open the file

input_path = os.path.abspath(input_tmp)
filename = input_path.split("/")[-1]
pattern_name = r'(\w+)_run_(\w+)_ov_(\d+).00_sipmgr_(\d+)_(\w+)'
name_match = re.match(pattern_name, filename)
if name_match:
    run = str(name_match.group(2))
    ov = int(name_match.group(3))
    channel = int(name_match.group(4))
    sipm_type = name_match.group(5)

if not os.path.isdir(output_path + "/pdf"):
    os.makedirs(output_path + "/pdf")
##########################################################
df = pd.read_csv(f'{csv_path}/charge_fit_tile_ch{channel}_ov{ov}.csv')
f1 = ROOT.TFile(input_path)
tree = f1.Get(tree_name)
for po in range(16):
    gain = df.loc[  (df['channel'] == channel) & (df['position'] == po) & (df['voltage'] == ov)].head(1)["gain"].values[0]
    events = df.loc[  (df['channel'] == channel) & (df['position'] == po) & (df['voltage'] == ov)].head(1)["events"].values[0]
    sigma1_value = df.loc[  (df['channel'] == channel) & (df['position'] == po) & (df['voltage'] == ov)].head(1)["sigma1"].values[0]
    sigma2_value = df.loc[  (df['channel'] == channel) & (df['position'] == po) & (df['voltage'] == ov)].head(1)["sigma2"].values[0]
    bkgPDF = ROOT.TH1F("bkgPDF","bkgPDF",ov_ranges[ov], -20, ov_ranges[ov] - 20)#ROOT.gPad.GetPrimitive("bkgPDF") 
    tree.Draw(f"bkgQ_ch{po}>>bkgPDF")
    #bkgPDF = ROOT.gPad.GetPrimitive("bkgPDF") 
    sigQ = ROOT.RooRealVar("sigQ", "sigQ", -20, ov_ranges[ov] - 20)
    bkghist = ROOT.RooDataHist("hist", "Data Histogram", ROOT.RooArgList(sigQ), ROOT.RooFit.Import(bkgPDF))
    pdf = ROOT.RooHistPdf("pdf", "Histogram PDF", ROOT.RooArgSet(sigQ), bkghist)
    mean1 = ROOT.RooRealVar("mean1", "mean1", gain)
    mean2 = ROOT.RooRealVar("mean2", "mean2", gain * 2)
    sigma1 = ROOT.RooRealVar("sigma1", "sigma1", sigma1_value, sigma1_value * 0.9, sigma1_value * 1.1)
    sigma2 = ROOT.RooRealVar("sigma2", "sigma2", sigma2_value, sigma2_value * 0.9, sigma2_value * 1.1)
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", sigQ, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", sigQ, mean2, sigma2)
    
    
    pdf_list = ROOT.RooArgList(pdf, gauss1,gauss2)  # List of PDFs
    
    coeff_list = ROOT.RooArgList()  # List of coefficients
    
    coeff1 = ROOT.RooRealVar("coeff1", "coeff1", 0.5, 0, 1)
    coeff2 = ROOT.RooRealVar("coeff2", "coeff2", 0.25, 0, 1)
    coeff3 = ROOT.RooFormulaVar("coeff3", "coeff3", "1-coeff1-coeff2", ROOT.RooArgList(coeff1, coeff2))
    
    for i in range(1,4):
        coeff_list.add(globals()[f'coeff{i}'])
    #add_pdf = ROOT.RooAddPdf("add_pdf", "Combined PDF", pdf_list)
    add_pdf = ROOT.RooAddPdf("add_pdf", "Combined PDF", pdf_list, coeff_list)
    hist = ROOT.TH1F("chargehist", "charge hist", ov_ranges[ov], -20, ov_ranges[ov] - 20)
    tree.Draw(f"{variable_name}_ch{po}>>chargehist")
    #hist = ROOT.gPad.GetPrimitive("chargehist")
    data = ROOT.RooDataHist("data","data", ROOT.RooArgSet(sigQ), hist)
    #mean = ROOT.RooRealVar("mean", "mean", bkgPDF.GetMean(), bkgPDF.GetMean() - bkgPDF.GetRMS(),bkgPDF.GetMean() + bkgPDF.GetRMS())
    #sigma = ROOT.RooRealVar("sigma", "sigma", bkgPDF.GetRMS(),  0.01, bkgPDF.GetRMS() * 2)
    #gauss = ROOT.RooGaussian( "gauss", "gauss", sigQ, mean, sigma)
    
    # Perform the fit
    #result = minimizer.fit("")
    #pdf.fitTo(data, ROOT.RooFit.Minimizer(minimizer), )
    result = add_pdf.fitTo(data, ROOT.RooFit.Minimizer("Minuit2"), ROOT.RooFit.Save())  # Pass the specified minimizer
    #result2 = gauss.fitTo(data, ROOT.RooFit.Minimizer("Minuit2"), ROOT.RooFit.Save())
    
    total_entries = data.sumEntries()
    
    pdf_norm = pdf.createIntegral(ROOT.RooArgSet(sigQ), ROOT.RooFit.NormSet(sigQ), ROOT.RooFit.Range("sigQ"))
    gauss1_norm = gauss1.createIntegral(ROOT.RooArgSet(sigQ), ROOT.RooFit.NormSet(sigQ), ROOT.RooFit.Range("sigQ"))
    gauss2_norm = gauss2.createIntegral(ROOT.RooArgSet(sigQ), ROOT.RooFit.NormSet(sigQ), ROOT.RooFit.Range("sigQ"))
    # Create a normalization object for the Gaussian PDF
    # Calculate the total number of events predicted by the signal PDF
    canvas = ROOT.TCanvas("c1","c1", 1200, 800)
    if po == 0:
        canvas.Print(f"last.pdf[")
    # Divide the canvas into two asymmetric pads
    pad1 =ROOT.TPad("pad1","This is pad1",0.05,0.05,0.72,0.97);
    pad2 = ROOT.TPad("pad2","This is pad2",0.72,0.05,0.98,0.97);
    pad1.Draw()
    pad2.Draw()
    pad1.cd()
    frame = sigQ.frame()
    frame.SetXTitle("Charge")
    frame.SetYTitle("Events")
    frame.SetTitle(f"DCR Charge spectrum fit of overvoltage (Run {run} ov {ov}V ch{channel} tile{po})")
    data.plotOn(frame)
    #theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
    #theWorkSpace.pdf("final").plotOn(frame, ROOT.RooFit.Components("background"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
    #pdf.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
    #scaled_gauss1 = ROOT.RooProduct("scaled_gauss1", "scaled_gauss1", ROOT.RooArgList(coeff2, gauss1))
    
    #n_gauss1 = ROOT.RooRealVar("n_gauss1", "n_gauss1", 0, 10000)
    #n_gauss1.setVal(coeff2.getVal() * data.sumEntries())
    #gauss1_ext = ROOT.RooExtendPdf("gauss1_ext", "gauss1 with coefficient", gauss1, n_gauss1)
    
    #scaled_gauss1.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
    #gauss1_ext.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kGreen))
    #gauss2.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kGreen))
    add_pdf.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))
    # Plot the individual PDFs with the yields from the fit
    add_pdf.plotOn(frame, ROOT.RooFit.Components("pdf"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kOrange))
    add_pdf.plotOn(frame, ROOT.RooFit.Components("gauss1"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kBlue))
    add_pdf.plotOn(frame, ROOT.RooFit.Components("gauss2"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen))
    
    #gauss.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))
    frame.Draw()
    # Create TLine objects for the legend
    pad2.cd()
    # Create a TPaveText to display parameter values and uncertainties
    param_box = ROOT.TPaveText(0.01, 0.9, 0.9, 0.1, "NDC")
    param_box.SetFillColor(ROOT.kWhite)
    param_box.SetBorderSize(1)
    param_box.SetTextFont(42)
    param_box.SetTextSize(0.08)
    param_box.AddText(f"total = {total_entries} ")
    param_box.AddText(f"e noise = {coeff1.getVal():.3f} ")
    param_box.AddText(f"gauss1 = {coeff2.getVal():.3f} ")
    param_box.AddText(f"gauss2 = {coeff3.getVal():.3f} ")
    param_box.AddText(f"DCR = {total_entries * (1 - coeff1.getVal()) / 144. / (1100 * 8e-9 * events)} ")

    param_box.AddText(f"#sigma1 = {sigma1.getVal():.3f} #pm {sigma1.getError():.3f}")
    param_box.AddText(f"#sigma2 = {sigma2.getVal():.3f} #pm {sigma2.getError():.3f}")
    param_box.Draw("same")
    canvas.Print("last.pdf")
canvas.Print("last.pdf]")
    
