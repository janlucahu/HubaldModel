import time
import numpy as np
from probability_distributions import half_normal


def calc_collision_probability(parameters1, parameters2, const1, const2, sigma, timestep, sinE, cosE):
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
    start = time.time()

    sinE1 = sinE2 = sinE
    cosE1 = cosE2 = cosE

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

    dist = np.linalg.norm(np.array([x1 - x2, y1 - y2, z1 - z2]), axis=0)

    minDistance = np.min(dist)

    if parameters1[6] != -1 and parameters2[6] != -1:
        monthsToSeconds = 30 * 24 * 60 * 60
        activeSatellite = parameters1[6] + parameters2[6]  # = false if both are inactive
        synodicPeriod = 1 / np.abs(1 / parameters1[5] - 1 / parameters2[5])
        numberOfApproaches = int(timestep * monthsToSeconds / synodicPeriod)

        if activeSatellite:
            colProbPerApproach = half_normal(minDistance, sigma, True)
        else:
            colProbPerApproach = half_normal(minDistance, sigma, False)

        colProb = 1 - (1 - colProbPerApproach) ** numberOfApproaches
    else:
        colProb = 0

    finish = time.time()
    elapsed_time = np.round(finish - start, 6)
    print(f"Finished after {elapsed_time}s")

    return colProb
