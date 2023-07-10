import ROOT
import math
from ROOT import RooRealVar, RooGenericPdf, RooArgSet, RooArgList, RooDataHist, RooFit, TCanvas, TH1F, TRandom3, RooFormulaVar, RooGaussian, RooAddPdf

# Define the parameters of the distribution
lambda_ = RooRealVar("lambda", "lambda", 0.2)
mu = RooRealVar("mu", "mu", 2)

# Create RooFit variables for the observables and parameters
sigQ = RooRealVar("sigQ", "sigQ", -20, 100)
sigma0 = RooRealVar("sigma0", "sigma0", 2,)
sigma1 = RooRealVar("sigma1", "sigma1", 2,)
sigma2 = RooRealVar("sigma2", "sigma2", 2,)
sigma3 = RooRealVar("sigma3", "sigma3", 2,)
sigma4 = RooRealVar("sigma4", "sigma4", 2,)
sigma5 = RooRealVar("sigma5", "sigma5", 2,)
# Define the Gaussian PDF
ped = RooRealVar("ped", "ped", 0, -2, 2)
gain = RooRealVar("gain", "gain", 15, 5, 30)
mu0 = RooFormulaVar("mu0", "ped",RooArgList(ped) )
mu1 = RooFormulaVar("mu1", "ped + gain",RooArgList(ped, gain) )
mu2 = RooFormulaVar("mu2", "ped + 2 * gain",RooArgList(ped, gain) )
mu3 = RooFormulaVar("mu3", "ped + 3 * gain",RooArgList(ped, gain) )
mu4 = RooFormulaVar("mu4", "ped + 4 * gain",RooArgList(ped, gain) )
mu5 = RooFormulaVar("mu5", "ped + 5 * gain",RooArgList(ped, gain) )
gauss0 = RooGaussian("gauss0", "gaussian PDF at peak 0", sigQ, mu0, sigma0)
gauss1 = RooGaussian("gauss1", "gaussian PDF at peak 1", sigQ, mu1, sigma1)
gauss2 = RooGaussian("gauss2", "gaussian PDF at peak 2", sigQ, mu2, sigma2)
gauss3 = RooGaussian("gauss3", "gaussian PDF at peak 3", sigQ, mu3, sigma3)
gauss4 = RooGaussian("gauss4", "gaussian PDF at peak 4", sigQ, mu4, sigma4)
gauss5 = RooGaussian("gauss5", "gaussian PDF at peak 5", sigQ, mu5, sigma5)
# Generic pdf with the Generalized Poisson distribution formula
# It's crucial to make sure this formula matches your specific distribution
#formula = "mu * pow(mu + x * lambda, x-1) / TMath::Factorial(x) * exp(-mu - x * lambda)"
# Create a list for Gaussians and their fractions
gauss_list = RooArgList()
coeff_list = RooArgList()
# Sum the Gaussians with the Generalized Poisson probabilities as coefficients
coeff_vars = [RooRealVar(f"coeff{i}", f"coeff{i}", 1, 0, 1) for i in range(6)]

for i in range(6):
    gauss = globals()[f"gauss{i}"]
    gauss_list.add(gauss)
    
    if i <= 5:  # Exclude the last coefficient which is calculated automatically
        if i == 0:
            coeff = RooFormulaVar(f"coeff_formula{i}", "exp(-mu)", RooArgList(mu))
        else:
            coeff = RooFormulaVar(f"coeff_formula{i}", f"mu * pow(mu + {i} * lambda, {i}-1) / TMath::Factorial({i}) * exp(-mu - {i} * lambda)", RooArgList(mu, lambda_))
        
        coeff_vars[i].setVal(coeff.getVal())
        coeff_list.add(coeff_vars[i])
print(gauss_list, coeff_list)
pdf_sum = RooAddPdf("pdf_sum", "pdf_sum", gauss_list, coeff_list)
# Generate some data
data = pdf_sum.generate(RooArgSet(sigQ), 10000)

# Fit the data
result = pdf_sum.fitTo(data, ROOT.RooFit.Save())
final_params = result.floatParsFinal()
mu_value = final_params.find("coeff_formula0").getVal()
print(math.log(mu_value))


# Plot the data and the fit
c = TCanvas("c", "c", 800, 600)
frame = sigQ.frame()
data.plotOn(frame)
pdf_sum.plotOn(frame)
frame.Draw()

c.SaveAs("fit.png")
