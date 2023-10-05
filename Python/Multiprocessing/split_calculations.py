# number of calculations: N ** 2 / 2 - N / 2
import numpy as np
from numba import jit
from probability_distributions import half_normal


@jit(nopython=True)
def find_minimum(parameters1, parameters2, const1, const2, acc=20):
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

    minDistance = np.min(dist)

    return minDistance


@jit(nopython=True)
def collision_probability(sat1, sat2, satParameters, satConstants, sigma, timestep, acc=20):
    if satParameters[sat1][6] != -1 and satParameters[sat2][6] != -1:
        monthsToSeconds = 30 * 24 * 60 * 60
        activeSatellite = satParameters[sat1][6] + satParameters[sat2][6]  # = false if both are inactive
        synodicPeriod = 1 / np.abs(1 / satParameters[sat1][5] - 1 / satParameters[sat2][5])
        numberOfApproaches = int(timestep * monthsToSeconds / synodicPeriod)
        closestDistance = find_minimum(satParameters[sat1], satParameters[sat2], satConstants[sat1],
                                       satConstants[sat2], acc=acc)
        if activeSatellite:
            colProbPerApproach = half_normal(closestDistance, sigma, True)
        else:
            colProbPerApproach = half_normal(closestDistance, sigma, False)

        colProb = 1 - (1 - colProbPerApproach) ** numberOfApproaches
    else:
        colProb = 0

    return colProb


#@jit(nopython=True)
def calculation_slices(satIndices, numberOfWorkers):
    numberOfCalculations = 0
    for ii, ind in enumerate(satIndices):
        numberOfCalculations += ind
        if ii != len(satIndices) - 1:
            numberOfCalculations += satIndices[ii + 1] - ind - 1
    print(f"Total number of calculations: {numberOfCalculations}")
    calculationsPerWorker = np.ceil(numberOfCalculations / numberOfWorkers)
    print(f"calculations per worker: {int(calculationsPerWorker)}")
    slices = []
    indices = []
    numCalculations = []
    sliceCalculations = 0
    for ii, ind in enumerate(satIndices):
        if sliceCalculations < calculationsPerWorker:
            indices.append(ind)
            sliceCalculations += ind
            if ii != len(satIndices) - 1:
                numberOfCalculations += satIndices[ii + 1] - ind - 1
            if ind == satIndices[-1]:
                slices.append(indices)
                numCalculations.append(sliceCalculations)
        else:
            slices.append(indices)
            numCalculations.append(sliceCalculations)
            sliceCalculations = 0
            indices = [ind]

    for ii, calculations in enumerate(numCalculations):
        print(f"Worker {ii + 1}: Calculations: {calculations}")
    return slices


@jit(nopython=True)
def sparse_prob_matrix(satParameters, satConstants, sigma, timestep, satIndices, acc=20):
    sparseProbList = []
    probThresh = 10 ** (-10)
    for sat1 in satIndices:
        for sat2 in range(sat1):
            colProb = collision_probability(sat1, sat2, satParameters, satConstants, sigma, timestep, acc)
            if colProb > probThresh:
                sparseProbList.extend([sat1, sat2, colProb])

    sparseProbMatrix = np.array(sparseProbList).reshape(-1, 3)

    return sparseProbMatrix
