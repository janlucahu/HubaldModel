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
        yy = 5 * 10 ** (-5) * np.exp(-xx ** 2 / (2 * sigma ** 2))
    else:
        yy = 5 * 10 ** (-1) * np.exp(-xx ** 2 / (2 * sigma ** 2))
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


@jit(nopython=True)
def linear_distribution(xx, mm, bb):
    yy = mm * xx + bb
    return yy


def plot_distributions(x1, sigma, active, x2, gg, dd, cc, x3, mm, bb, plot=(1, 1, 1)):
    plot_normal, plot_logistic, plot_linear = plot
    if plot_normal:
        hnx = np.linspace(0, x1, 1000)
        halfNormal = half_normal(hnx, sigma, active)
        plt.plot(hnx, halfNormal, 'r-', lw=2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Half Normal Distribution with Standard Deviation σ={sigma}')
        plt.show()

    if plot_logistic:
        lx = np.linspace(0, x2, 1000)
        logistic = logistic_distribution(lx, gg, dd, cc)
        plt.plot(lx, logistic, 'r-', lw=2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Logistic Distribution with d={dd}, c={cc}')
        plt.show()

    if plot_linear:
        linx = np.linspace(0, x3, 1000)
        linear = linear_distribution(linx, mm, bb)
        plt.plot(linx, linear, 'r-', lw=2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Linear Distribution with m={mm}, b={bb}')
        plt.show()


if __name__ == "__main__":
    x1 = 5000
    sigma = 2000
    active = 1

    x2 = 1000000000
    gg = 1
    dd = 10000
    cc = 0.00000002

    x3 = 1000000000
    mm = 1 / 10000 / 1000000000
    bb = 0

    plot_distributions(x1, sigma, active, x2, gg, dd, cc, x3, mm, bb, plot=(0, 0, 1))
