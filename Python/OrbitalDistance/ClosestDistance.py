import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


EMIN, EMAX = 0, 0.1
IMIN, IMAX = 0, 0.1 * np.pi
WMIN, WMAX = 0, 0.1 * np.pi
OMIN, OMAX = 0, 0.1 * np.pi
TMIN, TMAX = 1.5, 8


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
def initialize(nrOfSats, alimits, activeFraction, plane=False):
    satParameter = np.empty((nrOfSats, 7))
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
        if np.random.rand() < activeFraction:
            active = 1
        else:
            active = 0

        satParameter[satNr][0] = aa
        satParameter[satNr][1] = ee
        satParameter[satNr][2] = ii
        satParameter[satNr][3] = ww
        satParameter[satNr][4] = Om
        satParameter[satNr][5] = TT
        satParameter[satNr][6] = active

        satConstants[satNr] = constants(satParameter[satNr])

    return satParameter, satConstants


@jit(nopython=True)
def find_minimum(parameters1, parameters2, const1, const2, acc=100, repetitions=3):
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
    distances = np.zeros((nrOfSats, nrOfSats))

    for sat1 in range(nrOfSats):
        print(sat1 + 1, ' of ', nrOfSats)
        for sat2 in range(sat1):
            closestDistance = find_minimum(satParameters[sat1],
                                           satParameters[sat2],
                                           satConstants[sat1],
                                           satConstants[sat2], acc=acc)
            distances[sat1][sat2] = closestDistance
    return distances


@jit(nopython=True)
def probability_matrix(distanceMatrix, satParameters, sigma):
    colProbMatrix = np.zeros(distanceMatrix.shape)
    for sat1 in range(colProbMatrix.shape[0]):
        for sat2 in range(sat1):
            activeSatellite = satParameters[sat1][6] + satParameters[sat2][6]  # = false if both are inactive
            if activeSatellite:
                colProbMatrix[sat1][sat2] = half_normal(distanceMatrix[sat1][sat2], sigma, True)
            else:
                colProbMatrix[sat1][sat2] = half_normal(distanceMatrix[sat1][sat2], 2 * sigma, False)

    return colProbMatrix


@jit(nopython=True)
def fragment_collision(colProbMatrix, satParameters, satConstants, numberOfFragments, dd, cc, sigma, accuracy):
    satellitesStruck = 0
    activeSatParameters = satParameters[:, -1] == 1
    activeSatellites = np.count_nonzero(activeSatParameters)
    for ii in range(activeSatellites):
        pp = np.random.rand()
        fragmentCollisionProb = logistic_distribution(numberOfFragments, 1, dd, cc)
        if pp < fragmentCollisionProb:
            struckSat, act = np.where(satParameters == 1)
            randSat = struckSat[np.random.randint(0, len(struckSat))]
            satParameters[randSat][6] = 0
            print(f'Satellite {randSat} struck by fragment')
            satellitesStruck += 1

            for jj in range(randSat):
                activeSatellite = satParameters[jj][6]
                closestDistance = find_minimum(satParameters[randSat], satParameters[jj],
                                               satConstants[randSat], satConstants[jj], acc=accuracy)
                colProb = half_normal(closestDistance, sigma, activeSatellite)
                colProbMatrix[randSat][jj] = colProb
            for jj in range(len(colProbMatrix[randSat][randSat:-1])):
                ind = jj + randSat + 1
                activeSatellite = satParameters[ind][6]
                closestDistance = find_minimum(satParameters[randSat], satParameters[ind],
                                               satConstants[randSat], satConstants[ind], acc=accuracy)
                colProb = half_normal(closestDistance, sigma, activeSatellite)
                colProbMatrix[ind][randSat] = colProb

    return colProbMatrix, satParameters, satConstants, satellitesStruck


def satellite_collision(colProbMatrix, satParameters, satConstants, numberOfFragments, freeIndices, tt, tmax):
    pMatrix = np.random.rand(*colProbMatrix.shape)
    rows, cols = np.where(pMatrix < colProbMatrix)
    collisionsInIteration = len(rows)

    for ii in range(collisionsInIteration):
        numberOfFragments += 50_000
        sat1 = rows[ii]
        sat2 = cols[ii]
        freeIndices.append(sat1)
        freeIndices.append(sat2)
        print('*****************************************************************************************************')
        print(f'Collision between satellites {sat2} and {sat1},   '
              f'status: {satParameters[sat1][6]} {satParameters[sat2][6]},    '
              f'iteration {tt} of {tmax}')
        print('*****************************************************************************************************')
        satParameters[sat1][:] = np.array([0, 0, 0, 0, 0, 0, -1])
        satParameters[sat2][:] = np.array([0, 0, 0, 0, 0, 0, -1])
        satConstants[sat1][:] = np.array([0, 0, 0, 0, 0, 0])
        satConstants[sat2][:] = np.array([0, 0, 0, 0, 0, 0])

        # Change rows of collided satellites
        for jj in range(colProbMatrix.shape[0]):
            colProbMatrix[sat1][jj] = 0
        for jj in range(colProbMatrix.shape[0]):
            colProbMatrix[sat2][jj] = 0

        # Change columns of collided satellites
        for jj in range(colProbMatrix.shape[1]):
            colProbMatrix[jj][sat1] = 0
        for jj in range(colProbMatrix.shape[1]):
            colProbMatrix[jj][sat2] = 0
    return colProbMatrix, satParameters, satConstants, numberOfFragments, collisionsInIteration, freeIndices


def deorbit_and_launch(colProbMatrix, satParameters, satConstants, aLimits, accuracy, sigma, freeIndices):
    deorbitingSats = np.random.randint(0, 2)
    oldSats = []
    for oldSat in range(len(satParameters)):
        if satParameters[oldSat][6] == 0:
            oldSats.append(oldSat)
    if len(oldSats) > deorbitingSats:
        for ii in range(deorbitingSats):
            randIndex = np.random.randint(0, len(oldSats))
            randOldSat = oldSats[randIndex]
            del oldSats[randIndex]
            print(f'Deorbitation of satellite {randOldSat}')
            satParameters[randOldSat][:] = np.array([0, 0, 0, 0, 0, 0, -1])
            freeIndices.append(randOldSat)
            for jj in range(colProbMatrix.shape[0]):
                colProbMatrix[randOldSat][jj] = 0
            for jj in range(colProbMatrix.shape[1]):
                colProbMatrix[jj][randOldSat] = 0

    launchedSats = np.random.randint(0, 5)
    newPars, newCons = initialize(launchedSats, aLimits, 1, plane=False)
    launchIndices = []
    for ii in range(launchedSats):
        if len(freeIndices) > 0:
            newSat = freeIndices[0]
            del freeIndices[0]
            launchIndices.append(newSat)
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
            for jj in range(newSat):
                closestDistance = find_minimum(satParameters[newSat], satParameters[jj],
                                               satConstants[newSat], satConstants[jj], acc=accuracy)
                colProb = half_normal(closestDistance, sigma, True)
                colProbMatrix[newSat][jj] = colProb
            for jj in range(len(colProbMatrix[newSat][newSat:-1])):
                ind = jj + newSat + 1
                closestDistance = find_minimum(satParameters[newSat], satParameters[ind],
                                               satConstants[newSat], satConstants[ind], acc=accuracy)
                colProb = half_normal(closestDistance, sigma, True)
                colProbMatrix[ind][newSat] = colProb

        else:
            currentSatNr = satParameters.shape[0]
            launchIndices.append(currentSatNr)
            satParameters = np.vstack((satParameters, newPars[ii]))
            satConstants = np.vstack((satConstants, newCons[ii]))

            # Append a row of zeros to the bottom of the array
            colProbMatrix = np.vstack((colProbMatrix, np.zeros((1, currentSatNr))))
            # Append a column of zeros to the right of the array
            colProbMatrix = np.hstack((colProbMatrix, np.zeros((currentSatNr + 1, 1))))
            for jj in range(currentSatNr):
                closestDistance = find_minimum(satParameters[jj], satParameters[-1], satConstants[jj], satConstants[-1],
                                               acc=accuracy)
                colProb = half_normal(closestDistance, sigma, 1)
                colProbMatrix[jj][-1] = colProb
    if len(launchIndices) > 0:
        print(f'Launch of new satellites: {launchIndices}')

    return colProbMatrix, satParameters, satConstants, freeIndices


def collect_data(collectedData, tt, collisions, satParameters, numberOfFragments, counter):
    activeSatParameters = satParameters[:, -1] == 1
    inactiveSatParameters = satParameters[:, -1] == 0
    activeSatellites = np.count_nonzero(activeSatParameters)
    inactiveSatellites = np.count_nonzero(inactiveSatParameters)
    totalSatellites = activeSatellites + inactiveSatellites

    collectedData[0][counter] = tt
    collectedData[1][counter] = collisions
    collectedData[2][counter] = np.sum(collectedData[1])
    collectedData[3][counter] = totalSatellites
    collectedData[4][counter] = activeSatellites
    collectedData[5][counter] = inactiveSatellites
    collectedData[6][counter] = numberOfFragments

    return collectedData


def plot_data(simulationData):
    tt = simulationData[0]
    collisionsPerIteration = simulationData[1]
    totalCollisions = simulationData[2]
    totalSatellites = simulationData[3]
    activeSatellites = simulationData[4]
    inactiveSatellites = simulationData[5]
    numberOfFragments = simulationData[6]

    plt.plot(tt, collisionsPerIteration)
    plt.xlabel('Collisions per Iteration')
    plt.ylabel('Time')
    plt.title('Collisions per Iteration')
    plt.show()

    plt.plot(tt, totalCollisions)
    plt.xlabel('Total collisions')
    plt.ylabel('Time')
    plt.title('Collisions over time')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(tt, activeSatellites, label='active')
    ax.plot(tt, inactiveSatellites, label='inactive')
    ax.plot(tt, totalSatellites, label='total')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of satellites')
    ax.set_title('Active and inactive satellites over time')
    ax.legend()
    plt.show()

    plt.plot(tt, numberOfFragments)
    plt.xlabel('Number of fragments')
    plt.ylabel('Time')
    plt.title('Fragments over time')
    plt.show()


def hubald_model(startingSats, tmax, timestep, aLimits=(200_000, 2_000_000), accuracy=20):
    sigma = 2000
    activePercentage = 0.8
    numberOfFragments = 1_000_000
    collectedData = np.empty((7, tmax // timestep))
    freeIndices = []

    satParameters, satConstants = initialize(startingSats, aLimits, activePercentage, plane=False)
    distanceMatrix = distance_matrix(startingSats, satParameters, satConstants, acc=accuracy)
    colProbMatrix = probability_matrix(distanceMatrix, satParameters, sigma)

    counter = 0
    for tt in range(0, tmax, timestep):
        colProbMatrix, satParameters, satConstants, satsStruck = fragment_collision(colProbMatrix, satParameters,
                                                                                    satConstants, numberOfFragments,
                                                                                    10000, 0.0000005, sigma, accuracy)

        colArgs = (colProbMatrix, satParameters, satConstants, numberOfFragments, freeIndices, tt, tmax)
        colProbMatrix, satParameters, satConstants, numberOfFragments, cols, freeIndices = satellite_collision(*colArgs)

        colProbMatrix, satParameters, satConstants, freeIndices = deorbit_and_launch(colProbMatrix, satParameters,
                                                                                     satConstants, aLimits, accuracy,
                                                                                     sigma, freeIndices)
        nonZeroRows = satParameters[:, 0] != 0
        numberOfSatellites = np.count_nonzero(nonZeroRows)
        print(f'Number of satellites: {numberOfSatellites}    Iteration: {tt}')
        collectedData = collect_data(collectedData, tt, cols, satParameters, numberOfFragments, counter)
        counter += 1
        print()
    return collectedData


def main():
    start = time.time()
    simulationData = hubald_model(1000, 1200, 3)
    print(f'Number of collisions: {int(simulationData[2][-1])}')
    finish = time.time()
    print(f'Process finished after: {round(finish - start, 2)}s')

    plot_data(simulationData)


if __name__ == '__main__':
    main()
