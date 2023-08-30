import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def half_normal(xx, sigma, active):
    '''
    Half normal distribution used for satellite collision probability.

    Args:
        xx (float): X value for distribution. In this case satellite distance.
        sigma (float): Standard deviation.
        active (bool): Active status of satellite.

    Returns:
        yy (float): Y value of distribution for xx > 0.
    '''
    if active:
        yy = np.sqrt(2 / (100 * np.pi)) * np.exp(-xx ** 2 / (2 * sigma ** 2))
    else:
        yy = np.sqrt(2 / np.pi) * np.exp(-xx ** 2 / (2 * sigma ** 2))
    return yy * (xx >= 0)


@jit(nopython=True)
def logistic_distribution(xx, gg, dd, cc):
    '''
    Logistic distribution used for calculating collision probabilities with fragments.

    Args:
        xx (float): X value for logistic distribution.
        gg (float): Scaling parameter.
        dd (float): Scaling parameter.
        cc (float): Scaling parameter.

    Returns:
        yy (float): Y value of distribution.
    '''
    yy = gg / (1 + dd * np.exp(-cc * xx))
    return yy


def plot_distributions(x1, sigma, active, x2, gg, dd, cc):
    hnx = np.linspace(0, x1, 1000)
    halfNormal = half_normal(hnx, sigma, active)
    plt.plot(hnx, halfNormal, 'r-', lw=2)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Half Normal Distribution with Standard Deviation σ={sigma}')
    plt.show()

    lx = np.linspace(0, x2, 1000)
    logistic = logistic_distribution(lx, gg, dd, cc)
    plt.plot(hnx, logistic, 'r-', lw=2)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Half Normal Distribution with Standard Deviation σ={sigma}')
    plt.show()


if __name__ == "__main__":
    x1 = 5000
    sigma = 250
    active = 1

    x2 = 10000000
    gg = 1
    dd = 1000000
    cc = 0.0000001
    plot_distributions(x1, sigma, active, x2, gg, dd, cc)
    print("Done")
