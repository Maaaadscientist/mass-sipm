import os, sys
import math

args = sys.argv[1:]
prob = float(args[1]) /float(args[0])
prob_err = (float(args[1]) + float(args[2])) /float(args[0])
mu = -math.log(prob)
mu_err = abs(-math.log(prob_err) + math.log(prob) )
print(mu, mu_err)
