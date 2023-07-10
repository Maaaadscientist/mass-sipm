from scipy.optimize import fsolve
from math import exp

def equation(lambda_val, p):
    return lambda_val * lambda_val * 0.5 * exp(-lambda_val) - p

# Known value of p
p = 0.5

# Solve the equation using fsolve
lambda_val = fsolve(equation, 1, args=(p,))

# Print the result
print("Lambda value:", lambda_val[0])

