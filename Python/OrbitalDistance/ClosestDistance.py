import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import PlotDistance


EMIN, EMAX = 0, 0.3
IMIN, IMAX = 0, np.pi
WMIN, WMAX = 0, 2 * np.pi
OMIN, OMAX = 0, 2 * np.pi
MMIN, MMAX = 0, 1000


#@jit(nopython=True)
def summation(NN):
    val = 0
    for ii in range(NN):
        val = val + ii
    return val


#@jit(nopython=True)
def initialize(nrOfSats, alimits, plane=False):
    '''
    Inititalizes a system of satellites orbiting around a focus point. The size
    of the system specifies the bounderies, which the satellites won't pass

    Args:
        nrOfSats (int): Number of satellites to be initialized
        size (int): Size of the system
        tmax (int): Maximum time, influencing the orbital period
        accuracy (int): Number of decimals to be rounded to. Defaults to 1.
        plane (bool): Choose between a plane orbit or 3d orbit

    Returns:
        satParameter (2darray): Orbital parameters for each Satellite. Columns
                                depict satellite number, rows orbital
                                parameters.
        satPositions (2darray): Positions for each satellite. Columns depict
                                satellite number, rows positional components.

    '''
    satParameter = np.empty((nrOfSats, 5))

    AMIN, AMAX = alimits

    for satNr in range(nrOfSats):
        ee = np.random.uniform(EMIN, EMAX)
        # upper limit assures that no satellite is out of bounds
        aa = np.random.uniform(AMIN, AMAX)

        if plane:
            ii = 0
        else:
            ii = np.random.uniform(IMIN, IMAX)

        ww = np.random.uniform(WMIN, WMAX)
        Om = np.random.uniform(OMIN, OMAX)

        satParameter[satNr][0] = aa
        satParameter[satNr][1] = ee
        satParameter[satNr][2] = ii
        satParameter[satNr][3] = ww
        satParameter[satNr][4] = Om

    return satParameter


#@jit(nopython=True)
def constants(parameters1, parameters2):
    i1, i2 = parameters1[2], parameters2[2]
    w1, w2 = parameters1[3], parameters2[3]
    O1, O2 = parameters1[4], parameters2[4]

    P11_1 = np.cos(O1) * np.cos(w1) - np.sin(O1) * np.cos(i1) * np.sin(w1)
    P12_1 = - np.cos(O1) * np.sin(w1) - np.sin(O1) * np.cos(i1) * np.cos(w1)
    P21_1 = np.sin(O1) * np.cos(w1) + np.cos(O1) * np.cos(i1) * np.sin(w1)
    P22_1 = - np.sin(O1) * np.sin(w1) + np.cos(O1) * np.cos(i1) * np.cos(w1)
    P31_1 = np.sin(i1) * np.sin(w1)
    P32_1 = np.sin(i1) * np.cos(w1)

    P11_2 = np.cos(O2) * np.cos(w2) - np.sin(O2) * np.cos(i2) * np.sin(w2)
    P12_2 = - np.cos(O2) * np.sin(w2) - np.sin(O2) * np.cos(i2) * np.cos(w2)
    P21_2 = np.sin(O2) * np.cos(w2) + np.cos(O2) * np.cos(i2) * np.sin(w2)
    P22_2 = - np.sin(O2) * np.sin(w2) + np.cos(O2) * np.cos(i2) * np.cos(w2)
    P31_2 = np.sin(i2) * np.sin(w2)
    P32_2 = np.sin(i2) * np.cos(w2)

    const1 = (P11_1, P12_1, P21_1, P22_1, P31_1, P32_1)
    const2 = (P11_2, P12_2, P21_2, P22_2, P31_2, P32_2)

    return const1, const2


#@jit(nopython=True)
def find_minimum(parameters1, parameters2, acc=100, repetitions=3):
    E_1 = np.linspace(0, 2 * np.pi, acc)
    E_2 = np.linspace(0, 2 * np.pi, acc)

    E1, E2 = np.empty((acc, acc)), np.empty((acc, acc))
    for kk in range(acc):
        for ll in range(acc):
            E1[kk][ll] = E_1[ll]
            E2[kk][ll] = E_2[kk]
    sinE1, cosE1 = np.sin(E1), np.cos(E1)
    sinE2, cosE2 = np.sin(E2), np.cos(E2)

    a1, a2 = parameters1[0], parameters2[0]
    e1, e2 = parameters1[1], parameters2[1]

    const1, const2 = constants(parameters1, parameters2)
    P11_1, P12_1, P21_1, P22_1, P31_1, P32_1 = const1
    P11_2, P12_2, P21_2, P22_2, P31_2, P32_2 = const2

    X1 = a1 * (cosE1 - e1)
    Y1 = a1 * np.sqrt(1 - e1 ** 2) * sinE1

    x1 = X1 * P11_1 + Y1 * P12_1
    y1 = X1 * P21_1 + Y1 * P22_1
    z1 = X1 * P31_1 + Y1 * P32_1

    X2 = a2 * (cosE2 - e2)
    Y2 = a2 * np.sqrt(1 - e2 ** 2) * sinE2

    x2 = X2 * P11_2 + Y2 * P12_2
    y2 = X2 * P21_2 + Y2 * P22_2
    z2 = X2 * P31_2 + Y2 * P32_2

    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    minRow = None
    minCol = None
    minDistance = None
    for rep in range(repetitions):
        if minRow is None and minCol is None:
            minDistance = round(np.min(dist), 2)
        else:
            ival = 2 / (10 ** rep)
            E_1 = np.linspace(E_1[minCol] - ival, E_1[minCol] + ival, acc)
            E_2 = np.linspace(E_2[minRow] - ival, E_2[minRow] + ival, acc)
            E1, E2 = np.empty((acc, acc)), np.empty((acc, acc))
            for kk in range(acc):
                for ll in range(acc):
                    E1[kk][ll] = E_1[ll]
                    E2[kk][ll] = E_2[kk]
            X1 = a1 * (np.cos(E1) - e1)
            Y1 = a1 * np.sqrt(1 - e1 ** 2) * np.sin(E1)

            x1 = X1 * P11_1 + Y1 * P12_1
            y1 = X1 * P21_1 + Y1 * P22_1
            z1 = X1 * P31_1 + Y1 * P32_1

            X2 = a2 * (np.cos(E2) - e2)
            Y2 = a2 * np.sqrt(1 - e2 ** 2) * np.sin(E2)

            x2 = X2 * P11_2 + Y2 * P12_2
            y2 = X2 * P21_2 + Y2 * P22_2
            z2 = X2 * P31_2 + Y2 * P32_2

            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

            minDistance = round(np.min(dist), 2)

    return minDistance


#@jit(nopython=True)
def distance_matrix(nrOfSats, satParameters, acc=20):
    nrOfDistances = int(1 / 2 * nrOfSats * (nrOfSats - 1))
    distances = np.empty(nrOfDistances)

    for sat1 in range(nrOfSats):
        print(sat1, ' of ', nrOfSats)
        for sat2 in range(sat1):
            closestDistance = find_minimum(satParameters[sat1],
                                           satParameters[sat2], acc=acc)

            index = summation(sat1) + sat2 - 1

            distances[index] = closestDistance

    return distances


def distance_histogram(distanceMatrix, bins=50):
    counts, bins = np.histogram(distanceMatrix, bins=bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel('Closest approach distance in m')
    plt.ylabel('Number of occurrence')
    plt.show()


def main():
    satellites = 1000
    alimits = (200_000, 2_000_000)
    accuracy = 20

    start = time.time()
    parameters = initialize(satellites, alimits, plane=False)
    distanceMatrix = distance_matrix(satellites, parameters, acc=accuracy)
    finish = time.time()

    print('Process finished after: ', finish - start)
    print('Mean distance: ', np.mean(distanceMatrix))

    entries = 5
    indices = np.argpartition(distanceMatrix, entries)[:entries]
    lowest_values = distanceMatrix[indices]
    print(f'Values of the lowest {entries} entries: ', lowest_values)

    distance_histogram(distanceMatrix)

    plot_orbits(par[0:10])


if __name__ == '__main__':
    main()
