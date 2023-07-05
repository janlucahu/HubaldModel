import numpy as np
import matplotlib.pyplot as plt

# Define the half normal distribution function with peak at 0 and standard deviation sigma
def half_normal(x, sigma):
    y = np.sqrt(2/np.pi) * np.exp(-x**2/(2*sigma**2))
    return y * (x >= 0)

# Generate a random sample from the half normal distribution with standard deviation sigma=2
sigma = 3000
sample = np.abs(np.random.normal(scale=sigma, size=10000))

# Plot the probability density function (PDF) of the half normal distribution with standard deviation sigma=2
x = np.linspace(0, 10000, 1000)
pdf = half_normal(x, sigma)
plt.plot(x, pdf, 'r-', lw=2)

# Show the histogram of the sample
#plt.hist(sample, bins=50, density=True)

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Half Normal Distribution with Peak at 0 and Standard Deviation σ={sigma}')
plt.show()
