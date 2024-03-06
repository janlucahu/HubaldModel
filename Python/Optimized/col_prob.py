import numpy as np
import matplotlib.pyplot as plt


def sigma(r1, r2, m1, m2, v):
    G = 6.6743 * 10 ** (-11)
    sig = np.pi * (r1 + r2) ** 2 * (1 + (2 * (m1 + m2) * G / (r1  + r2)) / v ** 2)
    return sig


def col_probability(v, sigma, t, R, a, i, beta, q, p):
    NN = v * sigma * t / (2 * np.pi * R * a * np.sqrt((np.sin(i) ** 2 - np.sin(beta) ** 2) * (R - q) * (p - R)))
    return NN


if __name__ == '__main__':
    v = 36_000
    r1 = 1
    r2 = 0.1
    m1 = 500
    m2 = 1
    sigma = sigma(r1, r2, m1, m2, v)
    t = 1
    R = 800
    a = 1200
    i = np.linspace(np.pi / 32, np.pi / 1.1, 1000)
    beta = 0
    q = 200
    p = 1000
    N = col_probability(v, sigma, t, R, a, i, beta, q, p)

    plt.plot(i, N)
    plt.show()
