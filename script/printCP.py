import os,sys
import math

def get_coeff(i, k):
    if i == 0 and k == 0:
        return 1
    elif i == 0 and k > 0:
        return 0
    else:
        return math.factorial(k) * math.factorial(k-1)/(math.factorial(i) * math.factorial(i-1) * math.factorial(k-i))

def printCP(k):
    formula = ""
    for i in range (0,k+1):
        aLine = f"{get_coeff(i,k):0.0f} * pow(mu*( 1-lambda),{i})*pow(lambda, {k-i})"
        print(aLine)
        formula += aLine
        if i != k:
            formula += "+"
    print(formula)

#def printCPlatex(k):
#    for i in range (k+1):
#        print(f"{get_coeff(i,k):0.0f}\mu^{i}\cdot( 1-\lambda)^{i}\cdot\lambda^{k-i}")
def printCPlatex(k):
    for i in range(1,k+1):
        coeff = get_coeff(i, k)
        exponent_mu = f"^{i}" if i != 0 else ""  # Exclude exponent if i is 0
        exponent_lambda = f"^{k-i}" if (k-i) != 0 else ""  # Exclude exponent if k-i is 0
        
        
        if i <= 1:
            term = f"{coeff:.0f}\mu"
            term += "\cdot(1-\\lambda)"
        else:
            term = f"{coeff:.0f}\mu{exponent_mu}"
            term += f"\cdot(1-\\lambda){exponent_lambda}"
        
        if k-i <= 1:
            term += "\cdot\\lambda"
        else:
            term += f"\cdot\\lambda{exponent_lambda}"
        
        print(term)


printCP(int(sys.argv[1]))
printCPlatex(int(sys.argv[1]))
