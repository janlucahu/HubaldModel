import numpy as np
import matplotlib.pyplot as plt

# Define the half normal distribution function with peak at 0 and standard deviation sigma
def half_normal(x, sigma, active):
    if active:
        y = np.sqrt(2 / (100 * np.pi)) * np.exp(-x ** 2 / (2 * sigma ** 2))
    else:
        y = np.sqrt(2 /np.pi) * np.exp(-x ** 2 / (2 * sigma ** 2))
    return y * (x >= 0)


def logistic_distribution(x, g, d, c):
    return g / (1 + d * np.exp(-c * x))

# Generate a random sample from the half normal distribution with standard deviation sigma=2
sigma = 500
sample = np.abs(np.random.normal(scale=sigma, size=10000))

# Plot the probability density function (PDF) of the half normal distribution with standard deviation sigma=2
x = np.linspace(0, 10000000, 1000)
pdf = half_normal(x, sigma, active=True)
logistic = logistic_distribution(x, 1, 1000000, 0.0000001)
plt.plot(x, logistic, 'r-', lw=2)

# Show the histogram of the sample
#plt.hist(sample, bins=50, density=True)

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Half Normal Distribution with Peak at 0 and Standard Deviation σ={sigma}')
plt.show()
