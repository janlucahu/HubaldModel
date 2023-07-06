import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
#import PlotDistance


EMIN, EMAX = 0, 0.3
IMIN, IMAX = 0, np.pi
WMIN, WMAX = 0, 2 * np.pi
OMIN, OMAX = 0, 2 * np.pi
MMIN, MMAX = 0, 1000
TMIN, TMAX = 1.5 , 8


@jit(nopython=True)
def summation(NN):
    val = 0
    for ii in range(NN):
        val = val + ii
    return val


@jit(nopython=True)
def half_normal(x, sigma):
    y = np.sqrt(2/np.pi) * np.exp(-x**2/(2*sigma**2))
    return y * (x >= 0)


@jit(nopython=True)
def constants(parameters):
    i1 = parameters[2]
    w1 = parameters[3]
    O1 = parameters[4]

    P11_1 = np.cos(O1) * np.cos(w1) - np.sin(O1) * np.cos(i1) * np.sin(w1)
    P12_1 = - np.cos(O1) * np.sin(w1) - np.sin(O1) * np.cos(i1) * np.cos(w1)
    P21_1 = np.sin(O1) * np.cos(w1) + np.cos(O1) * np.cos(i1) * np.sin(w1)
    P22_1 = - np.sin(O1) * np.sin(w1) + np.cos(O1) * np.cos(i1) * np.cos(w1)
    P31_1 = np.sin(i1) * np.sin(w1)
    P32_1 = np.sin(i1) * np.cos(w1)

    entries = [P11_1, P12_1, P21_1, P22_1, P31_1, P32_1]
    const = np.array(entries)

    return const


@jit(nopython=True)
def initialize(nrOfSats, alimits, plane=False):
    '''
    Initializes a system of satellites orbiting around a focus point. The size
    of the system specifies the boundaries, which the satellites won't pass

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
    satParameter = np.empty((nrOfSats, 6))
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
        TT = sign * np.random.uniform(TMIN, TMAX)

        satParameter[satNr][0] = aa
        satParameter[satNr][1] = ee
        satParameter[satNr][2] = ii
        satParameter[satNr][3] = ww
        satParameter[satNr][4] = Om
        satParameter[satNr][5] = TT

        satConstants[satNr] = constants(satParameter[satNr])

    return satParameter, satConstants


@jit(nopython=True)
def find_minimum(parameters1, parameters2, const1, const2, acc=100,
                 repetitions=3):

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


# @jit(nopython=True)
def distance_matrix(nrOfSats, satParameters, satConstants, acc=20, flat=True):
    nrOfDistances = int(1 / 2 * nrOfSats * (nrOfSats - 1))
    if flat:
        distances = np.empty(nrOfDistances)
    else:
        distances = np.zeros((nrOfSats, nrOfSats))
    index = 0

    for sat1 in range(nrOfSats):
        print(sat1 + 1, ' of ', nrOfSats)
        for sat2 in range(sat1):
            closestDistance = find_minimum(satParameters[sat1],
                                           satParameters[sat2],
                                           satConstants[sat1],
                                           satConstants[sat2], acc=acc)

            if flat:
                distances[index] = closestDistance
                index += 1
            else:
                distances[sat2][sat1] = closestDistance
    return distances


def distance_histogram(distanceMatrix, bins=50):
    counts, bins = np.histogram(distanceMatrix, bins=bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel('Closest approach distance in m')
    plt.ylabel('Number of occurrence')
    plt.show()


def hubald_model(startingSats, tmax, timestep, aLimits=(200_000, 2_000_000), accuracy=20, satsPerCol=3):
    sigma = 1000
    parameters, constants = initialize(startingSats, aLimits, plane=False)
    distanceMatrix = distance_matrix(startingSats, parameters, constants, acc=accuracy, flat=False)
    colProbMatrix = half_normal(distanceMatrix, sigma)
    zeroThresh = half_normal(0.1, sigma)
    nrOfCollisions = 0
    collisionArr = []
    timeArr = []
    for tt in range(0, tmax, timestep):
        pp = np.random.rand()
        rows, cols = np.where((pp < colProbMatrix) & (colProbMatrix < zeroThresh))

        collisionsInIteration = len(rows)
        nrOfCollisions += collisionsInIteration
        collisionArr.append(nrOfCollisions)
        timeArr.append(tt)
        for ii in range(collisionsInIteration):
            sat1 = rows[ii]
            sat2 = cols[ii]
            print(f'Collision between satellites {sat1} and {sat2}')
            probPrint = round(colProbMatrix[sat1][sat2]  * 100, 1)
            print(f'probability: {probPrint}%,    iteration: {tt} of {tmax}')

            newPars, newCons = initialize(satsPerCol, aLimits, plane=False)
            # Change rows of collided satellites
            for jj in range(sat1):
                parameters[sat1] = newPars[0]
                constants[sat1] = newCons[0]
                closestDistance = find_minimum(parameters[sat1], parameters[jj], constants[sat1], constants[jj],
                                               acc=accuracy)
                colProb = half_normal(closestDistance, sigma)
                colProbMatrix[jj][sat1] = colProb

            for jj in range(sat2):
                parameters[sat2] = newPars[1]
                constants[sat2] = newCons[1]
                closestDistance = find_minimum(parameters[sat2], parameters[jj], constants[sat2], constants[jj],
                                               acc=accuracy)
                colProb = half_normal(closestDistance, sigma)
                colProbMatrix[jj][sat2] = colProb

            # Change columns of collided satellites
            for jj in range(len(colProbMatrix[sat1][sat1:-1])):
                ind = jj + sat1
                closestDistance = find_minimum(parameters[sat1], parameters[ind], constants[sat1], constants[ind],
                                               acc=accuracy)
                colProb = half_normal(closestDistance, sigma)
                colProbMatrix[sat1][ind] = colProb

            for jj in range(len(colProbMatrix[sat2][sat2:-1])):
                ind = jj + sat2
                closestDistance = find_minimum(parameters[sat2], parameters[ind], constants[sat2], constants[ind],
                                               acc=accuracy)
                colProb = half_normal(closestDistance, sigma)
                colProbMatrix[sat2][ind] = colProb


            for jj in range(satsPerCol - 2):
                currentSatNr = parameters.shape[0]
                parameters = np.vstack((parameters, newPars[jj + 2]))
                constants = np.vstack((constants, newCons[jj + 2]))

                # Append a row of zeros to the bottom of the array
                colProbMatrix = np.vstack((colProbMatrix, np.zeros((1, currentSatNr))))
                # Append a column of zeros to the right of the array
                colProbMatrix = np.hstack((colProbMatrix, np.zeros((currentSatNr + 1, 1))))
                for kk in range(currentSatNr):
                    closestDistance = find_minimum(parameters[kk], parameters[-1], constants[kk], constants[-1],
                                                   acc=accuracy)
                    colProb = half_normal(closestDistance, sigma)
                    colProbMatrix[kk][-1] = colProb
    return nrOfCollisions, collisionArr, timeArr


def main():
    start = time.time()
    colNr, collisionArr, timeArr = hubald_model(5000, 1000, 1)
    print(f'Number of collisions: {colNr}')
    finish = time.time()
    print(f'Process finished after: {finish - start}s')

    plt.plot(timeArr, collisionArr)
    plt.show()


if __name__ == '__main__':
    main()
