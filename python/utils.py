import math
import ROOT

status_descriptions = {
    0: "Converged successfully.",
    1: "Covariance matrix was made positive definite.",
    2: "Hesse matrix is invalid.",
    3: "Estimated distance to minimum too big.",
    4: "Maximum number of iterations exceeded.",
    5: "Maximum number of function calls exceeded.",
    6: "User requested termination.",
    7: "Function minimum is not valid.",
    8: "Up value is not valid."
}

def generalized_poisson(k, mu, lambda_):
    exp_term = f"TMath::Exp(-({mu} + {k} * {lambda_}))"
    main_term = f"{mu} * TMath::Power(({mu} + {k} * {lambda_}), {k}-1)"
    factorial_term = math.factorial(k)
    if type(k) is not int:
        raise ValueError("k should be an integer!") 
    return f"({main_term} * {exp_term} / {factorial_term})"

def poisson(k, mu):
    exp_term = f"TMath::Exp(-{mu})"
    main_term = f"TMath::Power({mu} , {k})"
    factorial_term = math.factorial(k)
    if type(k) is not int:
        raise ValueError("k should be an integer!") 
    if k >= 2:
        return f"({main_term} * {exp_term} / {factorial_term})"
    elif k == 0:
        return f"(TMath::Exp(-{mu}))"
    elif k == 1:
        return f"({mu} * TMath::Exp(-{mu}))"
    else:
        raise ValueError("k should not be negative")

def smeared_expo_convolution(x, x_k, beta, sigma_k):
    expo = f"TMath::Exp(-( {x} - {x_k}) / {beta}) / {beta}"
    cdf = f"(0.5 + 0.5 * TMath::Erf({x} - {x_k}/(TMath::Sqrt(2)*{sigma_k})))"
    return f"({expo} * {cdf}* (({x} > {x_k}) ? 1 : 0))"

def expo_convolution(x, x_k, beta, i):
    if i < 2 or type(i) is not int:
        raise ValueError("i should be an integer >= 2!")
    expo = f"TMath::Exp(-( {x} - {x_k}) / {beta}) / (TMath::Power({beta}, {i}) * {math.factorial(i-1)})"
    power = f"TMath::Power({x} - {x_k}, {i} - 1)"
    
    
    return f"({expo} * {power} * (({x} > {x_k}) ? 1 : 0))"

def binominalI(k, i, alpha):
    if k < 1 or type(k) is not int:
        raise ValueError("k should be an positive interger")
    if i < 0 or type(k) is not int:
        raise ValueError("i should be a non-negative interger")
    return f"(TMath::BinomialI({alpha}, {k}, {i}))"
    
#def gauss(x, x_k, sigma_k):
#    return f"(1/(TMath::Sqrt(2 * TMath::Pi())*{sigma_k} )* TMath::Exp(- TMath::Power({x} - {x_k}, 2)/(2 * TMath::Power({sigma_k}, 2))))"
def gauss(x, x_k, sigma_k):
    return f"(1/{sigma_k} * TMath::Exp(- TMath::Power({x} - {x_k}, 2)/(2 * TMath::Power({sigma_k}, 2))))"

def peak_pos(k, ped, gain):
    if k < 0 or type(k) is not int:
        raise ValueError("k should be a non-negative integer!")
    return f"({ped} + {k} * {gain})"

def prepare_pdf(n):
    if n < 1 or type(n) is not int:
        raise ValueError("k should be an positive interger")
    formula = ""
    GP0 = generalized_poisson(0, "mu", "lambda")
    x_0 = peak_pos(0, "ped", "gain")
    Gauss0 = gauss("sigQ", x_0, "sigma0")
    formula += f"{GP0} * {Gauss0}"
    for k in range(1, n+1):
        GPk = generalized_poisson(k, "mu", "lambda")
        x_k = peak_pos(k, "ped", "gain")
        Gaussk = gauss("sigQ", x_k, f"sigma{k}")
        formula += f"+ {GPk} * {Gaussk}"
    return formula
    
    
#
#for k in range(1, 15):
#    GPk = generalized_poisson(k, "mu", "lambda")
#    x_k = peak_pos(k, "ped", "gain")
#    Gaussk = gauss("sigQ", x_k, f"sigma{k}")
#    smeared_conv = smeared_expo_convolution("sigQ", x_k, "beta", f"sigma{k}") 
#    bino_0 = binominalI(k, 0, "alpha")
#    bino_1 = binominalI(k, 1, "alpha")
#    formula2 += f" + ( {GPk} * ({bino_0} * {Gaussk} + {bino_1} *{smeared_conv}"
#    if k >= 2:
#        for i in range(2, k):
#            bino_i = binominalI(k, i, "alpha")
#            normal_conv = expo_convolution("sigQ", x_k, "beta", i)
#            formula2 += f" + {bino_i} * {normal_conv}"
#    formula2 += "))"
#        
#    #formula2 += f" + {GPk} * {Gaussk}"
#    
#print(formula2)

formula2 = ""
GP0 = generalized_poisson(0, "mu", "lambda")
x_0 = peak_pos(0, "ped", "gain")
Gauss0 = gauss("sigQ", x_0, "sigma0")
formula2 += f"{GP0} * {Gauss0}"

for k in range(1, 15):
    GPk = generalized_poisson(k, "mu", "lambda")
    x_k = peak_pos(k, "ped", "gain")
    Gaussk = gauss("sigQ", x_k, f"sigma{k}")
    smeared_conv = smeared_expo_convolution("sigQ", x_k, "beta", f"sigma{k}") 
    bino_0 = binominalI(k, 0, "alpha")
    bino_1 = binominalI(k, 1, "alpha")
    formula2 += f" + ( {GPk} * ({bino_0} * {Gaussk} + {bino_1} *{smeared_conv}"
    if k >= 2:
        for i in range(2, k):
            bino_i = binominalI(k, i, "alpha")
            normal_conv = expo_convolution("sigQ", x_k, "beta", i)
            formula2 += f" + {bino_i} * {normal_conv}"
    formula2 += "))"
        
    #formula2 += f" + {GPk} * {Gaussk}"
with open("script/compound_pdf.py","w") as file0:
    file0.write("compound_pdf_str = ")
    file0.write("'") 
    file0.write(formula2)
    file0.write("'") 
print(formula2)
## Test the function
#mu=1.59361 
#lambda_ = 0.164561
#mean = 0
#mean_p = 0
#
#gp = 0
#for k in range(0, 20):
#    formula_string = generalized_poisson(k, mu, lambda_)
#    poison_string = poisson(k, mu)
#    f2 = ROOT.TF1("f", formula_string)
#    fp = ROOT.TF1("fp", poison_string)
#    result = f2.Eval(1000.)  
#    result_p = fp.Eval(0.)
#    gp += result
#    mean += k * result
#    mean_p += k * result_p
#    print(k, result, result_p)
#print(mean)
#print(mean_p)
#print(gp)
#    
#def calculate_GP(k, mu, lambda_):
#    formula_string = generalized_poisson(k, mu, lambda_)
#    f1 = ROOT.TF1("f", formula_string)
#    result = f1.Eval(1000.)  
#    return result
#for k in range(15):
#    print(f"calculate GP for {k}:", calculate_GP(k, 1.863, 0.172))
# 
##
##print(f"Value of the expression for mu = {mu_value} is {result}")
##x, x_k, beta, sigma_k = 'x', 0, 10., 5.
##cov_string = smeared_expo_convolution(x, x_k, beta, sigma_k)
##print(cov_string)
##f2 = ROOT.TF1("f2", cov_string, 0.0, 20)
##result = f2.Eval(1)
##print(result)
##
##x, x_k, beta, i = 'x', 0, 10., 5
##str4 = expo_convolution(x, x_k, beta, i)
##print(str4)
##f4 = ROOT.TF1("f4", str4, 0.0, 20)
##result = f4.Eval(10)
##print("str4",result)
##
##
##k, i, alpha =2, 1, "x"
##str3 = binominall(k, i, alpha)
##print(str3)
##f3 = ROOT.TF1("f2", str3, 0.0, 20)
##result = f3.Eval(0.1)
##print(result)
##
##
##x, x_k, sigma_k = 'x', 0, 1
##str5 = gauss(x, x_k, sigma_k)
##print(str5)
##f5 = ROOT.TF1("f5", str5, 0.0, 20)
##result = f5.Eval(-0.1)
##print("str5",result)
