import os, sys
import math

args = sys.argv[1:]
prob = float(args[1]) /float(args[0])
prob_err = (float(args[1]) + float(args[2])) /float(args[0])
mu = -math.log(prob)
mu_err = abs(-math.log(prob_err) + math.log(prob) )
events = float(args[0]) * math.exp(-mu) * mu
lambda_ = -math.log(float(args[3])/events)
lambda_err = abs(-math.log(float(args[3])/events)+ math.log((float(args[3])+float(args[4]))/events))
print(mu, mu_err)
print(lambda_, lambda_err)
