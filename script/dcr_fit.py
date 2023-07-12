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
bkgPDF = ROOT.TH1F("bkgPDF","bkgPDF",110, -10, 100)#ROOT.gPad.GetPrimitive("bkgPDF") 
tree.Draw(f"bkgQ_ch{po}>>bkgPDF")
#bkgPDF = ROOT.gPad.GetPrimitive("bkgPDF") 
sigQ = ROOT.RooRealVar("sigQ", "sigQ", -10, 100)
bkghist = ROOT.RooDataHist("hist", "Data Histogram", ROOT.RooArgList(sigQ), ROOT.RooFit.Import(bkgPDF))
pdf = ROOT.RooHistPdf("pdf", "Histogram PDF", ROOT.RooArgSet(sigQ), bkghist)
mean1 = ROOT.RooRealVar("mean1", "mean1", 20, 10, 50)
mean2 = ROOT.RooRealVar("mean2", "mean2", 45, 30, 50)
sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 5, 1, 10)
sigma2 = ROOT.RooRealVar("sigma2", "sigma2", 5, 1, 10)
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
hist = ROOT.TH1F("chargehist", "charge hist", 500, -10, 100)
tree.Draw(f"{branch_name}>>chargehist")
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


# Create a normalization object for the Gaussian PDF
# Calculate the total number of events predicted by the signal PDF
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
add_pdf.plotOn(frame, ROOT.RooFit.Components("pdf"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kMagenta))
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
param_box.Draw("same")
canvas.SaveAs("last.pdf")

