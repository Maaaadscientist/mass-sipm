import scipy.stats as stats

# Parameters of the Gaussian distribution
mean = 0
sigma = 3.579

# Define the range over which to calculate the normalization factor
lower_bound = -21.454/2
upper_bound = 21.454/2


# Calculate the probability using the CDF
probability = stats.norm.cdf(upper_bound, mean, sigma) - stats.norm.cdf(lower_bound, mean, sigma)

print("Probability within the range:", probability)
def get_full_gaussian_scale(mean, sigma, lower_bound, upper_bound):
    # Calculate the probability using the CDF
    probability = stats.norm.cdf(upper_bound, mean, sigma) - stats.norm.cdf(lower_bound, mean, sigma)
    return 1.0 / probability
