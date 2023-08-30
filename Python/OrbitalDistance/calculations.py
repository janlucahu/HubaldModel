import numpy as np
from numba import jit
from probability_distributions import *


EMIN, EMAX = 0, 0.3
IMIN, IMAX = 0, 0.5 * np.pi
WMIN, WMAX = 0, 1.0 * np.pi
OMIN, OMAX = 0, 1.0 * np.pi
TMIN, TMAX = 1.5, 8


@jit(nopython=True)
def constants(satParameters):
    '''
    Calculate constants used to calculate the orbits of the satellites.

    Args:
        satParameters (1darray): Orbital parameters of the satellites.

    Returns:
        satConstants (1darray): Rotation matrix elements for position calculation in satellite frame.
    '''
    i1 = satParameters[2]
    w1 = satParameters[3]
    O1 = satParameters[4]

    P11_1 = np.cos(O1) * np.cos(w1) - np.sin(O1) * np.cos(i1) * np.sin(w1)
    P12_1 = - np.cos(O1) * np.sin(w1) - np.sin(O1) * np.cos(i1) * np.cos(w1)
    P21_1 = np.sin(O1) * np.cos(w1) + np.cos(O1) * np.cos(i1) * np.sin(w1)
    P22_1 = - np.sin(O1) * np.sin(w1) + np.cos(O1) * np.cos(i1) * np.cos(w1)
    P31_1 = np.sin(i1) * np.sin(w1)
    P32_1 = np.sin(i1) * np.cos(w1)

    entries = [P11_1, P12_1, P21_1, P22_1, P31_1, P32_1]
    satConstants = np.array(entries)

    return satConstants


@jit(nopython=True)
def initialize(nrOfSats, alimits, activeFraction, plane=False):
    '''
    Initializes random orbital parameters for a given number of satellites.

    Args:
        nrOfSats (int): Number of satellites for which the orbital parameters should be initialized.
        alimits (float, float): Upper and lower limit for the semi-major axes.
        activeFraction (float): Percentage of active satellites initialized.
        plane (bool): Only planar orbits (inclination = 0) are initialized if set to True.

    Returns:
        satParameter (2darray): Orbital parameters, period time and active status for each satellite.
        satConstants (2darray): Rotation matrix elements for position calculation for each satellite.
    '''
    satParameters = np.empty((nrOfSats, 7))
    satConstants = np.empty((nrOfSats, 6))

    AMIN, AMAX = alimits
    signs = np.array([-1, 1])
    for satNr in range(nrOfSats):
        ee = np.random.uniform(EMIN, EMAX)
        aa = np.random.uniform(AMIN, AMAX)

        if plane:
            ii = 0
        else:
            ii = np.random.uniform(IMIN, IMAX)

        ww = np.random.uniform(WMIN, WMAX)
        Om = np.random.uniform(OMIN, OMAX)
        sign = signs[np.random.randint(0, 2)]
        CEarth = 9.91 * 10 ** (-14)
        TT = sign * np.sqrt(CEarth * aa ** 3)
        if np.random.rand() < activeFraction:
            active = 1
        else:
            active = 0

        satParameters[satNr][0] = aa
        satParameters[satNr][1] = ee
        satParameters[satNr][2] = ii
        satParameters[satNr][3] = ww
        satParameters[satNr][4] = Om
        satParameters[satNr][5] = TT
        satParameters[satNr][6] = active

        satConstants[satNr] = constants(satParameters[satNr])

    return satParameters, satConstants


@jit(nopython=True)
def find_minimum(parameters1, parameters2, const1, const2, acc=100, repetitions=3):
    '''
    Calculates the closest approach of two satellites. The minimum of the 2-dimensional distance function depending on
    the respective orbital anomalies is searched.

    Args:
        parameters1 (1darray): Set of orbital parameters for satellite 1.
        parameters2 (1darray): Set of orbital parameters for satellite 2.
        const1 (1darray): Rotation matrix elements for position calculation of satellite 1.
        const2 (1darray): Rotation matrix elements for position calculation of satellite 1.
        acc (int): Number of points per axes for 2-dimensional distance function.
        repetitions (int): For each repetition a smaller area is considered around the previously found minimum and a
                           more accurate minimum is found.

    Returns:
        minDistance (float): Closest approach distance between two satellites.
    '''
    E_1 = np.linspace(0, 2 * np.pi, acc)
    E_2 = np.linspace(0, 2 * np.pi, acc)

    E1 = E2 = np.empty((acc, acc))
    for kk in range(acc):
        for ll in range(acc):
            E1[kk][ll] = E_1[ll]
            E2[kk][ll] = E_2[kk]
    sinE1, cosE1 = np.sin(E1), np.cos(E1)
    sinE2, cosE2 = np.sin(E2), np.cos(E2)

    a1, a2 = parameters1[0], parameters2[0]
    e1, e2 = parameters1[1], parameters2[1]

    P11_1, P11_2 = const1[0], const2[0]
    P12_1, P12_2 = const1[1], const2[1]
    P21_1, P21_2 = const1[2], const2[2]
    P22_1, P22_2 = const1[3], const2[3]
    P31_1, P31_2 = const1[4], const2[4]
    P32_1, P32_2 = const1[5], const2[5]

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


@jit(nopython=True)
def distance_matrix(nrOfSats, satParameters, satConstants, acc=20):
    '''
    Computes a matrix containing the closest approach distance for each pair of satellites.

    Args:
        nrOfSats (int): Number of satellites and hence matrix size.
        satParameters (2darray): Orbital parameters, period time and active status for each satellite.
        satConstants (2darray): Rotation matrix elements for position calculation for each satellite.
        acc (int): Number of points per axes for 2-dimensional distance function.

    Returns:
        distanceMatrix (2darray): Matrix containing closest approach distances for each pair of satellites. Only the
                                  lower left triangle is filled.
    '''
    distanceMatrix = np.zeros((nrOfSats, nrOfSats))

    for sat1 in range(nrOfSats):
        print(sat1 + 1, ' of ', nrOfSats)
        for sat2 in range(sat1):
            closestDistance = find_minimum(satParameters[sat1],
                                           satParameters[sat2],
                                           satConstants[sat1],
                                           satConstants[sat2], acc=acc)
            distanceMatrix[sat1][sat2] = closestDistance
            if closestDistance < 10000:
                print(closestDistance)
    return distanceMatrix


@jit(nopython=True)
def probability_matrix(distanceMatrix, satParameters, sigma, timestep):
    '''
    Converts a matrix containing the closest approach distances into a matrix containing the collision probabilities for
    each pair of satellites. For probability calculation a half-normal distribution is used.

    Args:
        distanceMatrix (2darray): Matrix containing closest approach distances for each pair of satellites. Only the
                                  lower left triangle is filled.
        satParameters (2darray): Orbital parameters, period time and active status for each satellite.
        sigma (float): Standard deviation of half-normal distribution.

    Returns:
        colProbMatrix (2darray): Matrix containing collision probabilities for each pair of satellites. Only the lower
                                 left triangle is filled.
    '''
    colProbMatrix = np.zeros(distanceMatrix.shape)
    monthsToSeconds = 30 * 24 * 60 * 60
    for sat1 in range(colProbMatrix.shape[0]):
        for sat2 in range(sat1):
            activeSatellite = satParameters[sat1][6] + satParameters[sat2][6]  # = false if both are inactive
            synodicPeriod = 1 / np.abs(1/satParameters[sat1][5] - 1/satParameters[sat2][5])
            numberOfApproaches = int(timestep * monthsToSeconds / synodicPeriod)
            if activeSatellite:
                colProbPerApproach = half_normal(distanceMatrix[sat1][sat2], sigma, True)
                colProb = 1 - (1 - colProbPerApproach) ** numberOfApproaches
                colProbMatrix[sat1][sat2] = colProb
            else:
                colProbPerApproach = half_normal(distanceMatrix[sat1][sat2], sigma, False)
                colProb = 1 - (1 - colProbPerApproach) ** numberOfApproaches
                colProbMatrix[sat1][sat2] = colProb
    return colProbMatrix
