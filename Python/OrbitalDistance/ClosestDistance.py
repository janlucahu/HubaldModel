import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


EMIN, EMAX = 0, 0.3
IMIN, IMAX = 0, 0.5 * np.pi
WMIN, WMAX = 0, 1.0 * np.pi
OMIN, OMAX = 0, 1.0 * np.pi
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
        TT = sign * np.random.uniform(TMIN, TMAX)
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
    return distanceMatrix


@jit(nopython=True)
def probability_matrix(distanceMatrix, satParameters, sigma):
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
    for sat1 in range(colProbMatrix.shape[0]):
        for sat2 in range(sat1):
            activeSatellite = satParameters[sat1][6] + satParameters[sat2][6]  # = false if both are inactive
            if activeSatellite:
                colProbMatrix[sat1][sat2] = half_normal(distanceMatrix[sat1][sat2], sigma, True)
            else:
                colProbMatrix[sat1][sat2] = half_normal(distanceMatrix[sat1][sat2], 2 * sigma, False)

    return colProbMatrix


@jit(nopython=True)
def small_fragment(colProbMatrix, satParameters, satConstants, smallFragments, dd, cc, sigma, accuracy):
    '''
    Updates the collision probability matrix considering the chance that a number of active satellites are hit by small
    fragments and become an inactive one. The probability of a fragment colliding with a satellite is calculated using a
    logistic distribution.

    Args:
        colProbMatrix (2darray): Matrix containing collision probabilities for each pair of satellites.
        satParameters (2darray): Orbital parameters, period time and active status for each satellite.
        satConstants (2darray): Rotation matrix elements for position calculation for each satellite.
        smallFragments (int): Total number of small fragments present.
        dd (float): Scaling parameter of logistic distribution.
        cc (float): Scaling parameter of logistic distribution.
        sigma (float): Standard deviation of half-normal distribution used for satellite collision probability.
        accuracy (int): Number of points per axes for 2-dimensional distance function.

    Returns:
        colProbMatrix (2darray): Updated collision probability matrix.
        satParameters (2darray): Updated satellite parameters with changed active statuses.
        satellitesStruck (int): Number of satellites hit by fragment.

    '''
    satellitesStruck = 0
    activeSatParameters = satParameters[:, -1] == 1
    activeSatellites = np.count_nonzero(activeSatParameters)
    for ii in range(activeSatellites):
        pp = np.random.rand()
        fragmentCollisionProb = logistic_distribution(smallFragments, 1, dd, cc)
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

    return colProbMatrix, satParameters, satellitesStruck


def large_fragment(colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, dd, cc):
    '''
    Large fragments are able to destroy satellites, setting their collision probability to 0, aswell as their
    parameters. Additionally, large fragment collision create small and large fragments.

    Args:
        colProbMatrix (2darray): Matrix containing collision probabilities for each pair of satellites.
        satParameters (2darray): Orbital parameters, period time and active status for each satellite.
        satConstants (2darray): Rotation matrix elements for position calculation for each satellite.
        smallFragments (int): Total number of small fragments present.
        largeFragments (int): Total number of large fragments present.
        freeIndices (list): Free indices in the collision probability matrix to be reused.
        dd (float): Scaling parameter of logistic distribution.
        cc (float): Scaling parameter of logistic distribution.

    Returns:
        colProbMatrix (2darray): Updated collision probability matrix.
        satParameters (2darray): Updated satellite parameters, destroyed satellite parameters are set to 0.
        satConstants (2darray): Updated satellite constants, destroyed satellite constants are set to 0.
        fragments (int, int): Number of small and large fragments.
        satellitesStruck (int): Number of satellites hit by fragment.
        freeIndices (list): Updated free indices in the collision probability matrix to be reused.
    '''
    satellitesStruck = 0
    # find indices of not destroyed satellites
    activity = satParameters[:, -1]
    mask = activity != -1
    nonDestroyed = np.where(mask)[0]
    for ii in range(len(nonDestroyed)):
        pp = np.random.rand()
        fragmentCollisionProb = logistic_distribution(largeFragments, 1, dd, cc)
        if pp < fragmentCollisionProb:
            randSat = nonDestroyed[np.random.randint(0, len(nonDestroyed))]
            satParameters[randSat][:] = np.array([0, 0, 0, 0, 0, 0, -1])
            satConstants[randSat][:] = np.array([0, 0, 0, 0, 0, 0])

            # Change rows of fragmented satellite
            for jj in range(colProbMatrix.shape[0]):
                colProbMatrix[randSat][jj] = 0

            # Change columns of fragmented satellite
            for jj in range(colProbMatrix.shape[1]):
                colProbMatrix[jj][randSat] = 0

            smallFragments += 50_000
            largeFragments += 200
            print(f'Satellite {randSat} struck by large fragment')
            satellitesStruck += 1
            freeIndices.append(randSat)
    fragments = (smallFragments, largeFragments)
    return colProbMatrix, satParameters, satConstants, fragments, satellitesStruck, freeIndices


def satellite_collision(colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices,
                        tt, tmax):
    '''
    Updates the collision probability matrix based on collisions between satellites. Two satellites can collide
    depending on their collision probability which will destroy them and set all probabilities in the relevant matrix
    entries to 0. Additionally, fragments are yielded by the collision raising the total number of fragments.
    Args:
        colProbMatrix (2darray): Matrix containing collision probabilities for each pair of satellites.
        satParameters (2darray): Orbital parameters, period time and active status for each satellite.
        satConstants (2darray): Rotation matrix elements for position calculation for each satellite.
        numberOfFragments (int): Total number of fragments.
        freeIndices (list): Free indices in the collision probability matrix to be reused.
        tt (int): Iteration step.
        tmax (int): Maximum iteration step.

    Returns:
        colProbMatrix (2darray): Updated collision probability matrix.
        satParameters (2darray): Updated satellite parameters, destroyed satellite parameters are set to 0.
        satConstants (2darray): Updated satellite constants, destroyed satellite constants are set to 0.
        numberOfFragments (int): Updated total number of fragments.
        collisionsInIteration (int): Number of collisions caused.
        freeIndices (list): Updated free indices in the collision probability matrix to be reused.
    '''
    pMatrix = np.random.rand(*colProbMatrix.shape)
    rows, cols = np.where(pMatrix < colProbMatrix)
    collisionsInIteration = len(rows)

    for ii in range(collisionsInIteration):
        smallFragments += 50_000
        largeFragments += 200
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
    fragments = (smallFragments, largeFragments)
    return colProbMatrix, satParameters, satConstants, fragments, collisionsInIteration, freeIndices


def deorbit_and_launch(colProbMatrix, satParameters, satConstants, aLimits, sigma, accuracy, freeIndices):
    '''
    Update collision probability matrix considering deorbitation of old satellites and launch of new ones.

    Args:
        colProbMatrix (2darray): Matrix containing collision probabilities for each pair of satellites.
        satParameters (2darray): Orbital parameters, period time and active status for each satellite.
        satConstants (2darray): Rotation matrix elements for position calculation for each satellite.
        aLimits (float, float): Lower and upper limit for the semi-major axes.
        sigma (float): Standard deviation of half-normal distribution used for satellite collision probability.
        accuracy (int): Number of points per axes for 2-dimensional distance function.
        freeIndices (list): Free indices in the collision probability matrix to be reused.

    Returns:
        colProbMatrix (2darray): Updated collision probability matrix.
        satParameters (2darray): Updated satellite parameters, destroyed satellite parameters are set to 0.
        satConstants (2darray): Updated satellite constants, destroyed satellite constants are set to 0.
        freeIndices (list): Updated free indices in the collision probability matrix to be reused.
    '''
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

    launchedSats = np.random.randint(0, 20)
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
    '''
    Collects various different quantities of the simulation of the kessler syndrome.

    Args:
        collectedData (2darray): Measured data.
        tt (int): Iteration step.
        collisions (int): Number of collisions.
        satParameters (2darray): Orbital parameters, period time and active status for each satellite.
        numberOfFragments (int): Total number of fragments.
        counter (int): Counter variable for indices assignment.

    Returns:
        collectedData (2darray): Updated measured data.
    '''
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
    '''
    Plots the gathered simulation data.
    Args:
        simulationData (2darray): Measured data.

    Returns:
        None.
    '''
    tt = simulationData[0]
    collisionsPerIteration = simulationData[1]
    totalCollisions = simulationData[2]
    totalSatellites = simulationData[3]
    activeSatellites = simulationData[4]
    inactiveSatellites = simulationData[5]
    numberOfFragments = simulationData[6]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(tt, collisionsPerIteration)
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Collisions per Iteration')
    axs[0, 0].set_title('Collisions per Iteration')

    axs[0, 1].plot(tt, totalCollisions)
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Total collisions')
    axs[0, 1].set_title('Collisions over time')

    axs[1, 0].plot(tt, activeSatellites, label='active')
    axs[1, 0].plot(tt, inactiveSatellites, label='inactive')
    axs[1, 0].plot(tt, totalSatellites, label='total')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Number of satellites')
    axs[1, 0].set_title('Active and inactive satellites over time')
    axs[1, 0].legend()

    axs[1, 1].plot(tt, numberOfFragments)
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Number of fragments')
    axs[1, 1].set_title('Fragments over time')

    plt.tight_layout()
    plt.show()


def hubald_model(startingSats, tmax, timestep, aLimits=(200_000, 2_000_000), accuracy=20):
    '''
    Simulates the development of the population of the simulated orbit after starting with a given amount of satellites.
    The dynamics of the model work as following: Satellites are categorized as either active or inactive, distinguished
    by a significantly lower probability for active satellites to collide. The probability of the collision of
    satellites depends on their closest approach distance, yielding them destroyed after colliding, which means they are
    removed from the simulation, leaving fragments. Fragments have a chance of hitting active satellites, leaving them
    inactive. Also, inactive satellites can deorbit, which also means they are removed, while new satellites can be
    launched, hence added to the simulation.

    Args:
        startingSats (int): Number of satellites at the beginning of the simulation.
        tmax (int): Maximum iteration steps.
        timestep (int): Stepsize of the iterations.
        aLimits (float, float): Lower and upper limit for the semi-major axes.
        accuracy (int): Number of points per axes for 2-dimensional distance function.

    Returns:
        collectedData (2darray): Various quantities measured for each iteration step.
    '''
    sigma = 2000
    activePercentage = 0.3
    smallFragments = 1_000_000
    largeFragments = 10_000
    collectedData = np.empty((7, tmax // timestep))
    freeIndices = []

    satParameters, satConstants = initialize(startingSats, aLimits, activePercentage, plane=False)
    distanceMatrix = distance_matrix(startingSats, satParameters, satConstants, acc=accuracy)
    colProbMatrix = probability_matrix(distanceMatrix, satParameters, sigma)

    counter = 0
    for tt in range(0, tmax, timestep):
        d, c = 1000000, 0.0000001
        colProbMatrix, satParameters, satsStruck = small_fragment(colProbMatrix, satParameters, satConstants,
                                                                  smallFragments, d, c, sigma, accuracy)

        d, c = 10000, 0.00001
        fragmentArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, d, c)
        colProbMatrix, satParameters, satConstants, fragments, satsStruck, freeIndices = large_fragment(*fragmentArgs)
        smallFragments, largeFragments = fragments

        colArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, tt, tmax)
        colProbMatrix, satParameters, satConstants, fragments, cols, freeIndices = satellite_collision(*colArgs)
        smallFragments, largeFragments = fragments

        colProbMatrix, satParameters, satConstants, freeIndices = deorbit_and_launch(colProbMatrix, satParameters,
                                                                                     satConstants, aLimits, sigma,
                                                                                     accuracy, freeIndices)
        nonZeroRows = satParameters[:, 0] != 0
        numberOfSatellites = np.count_nonzero(nonZeroRows)
        print(f'Number of satellites: {numberOfSatellites}    Iteration: {tt}')
        collectedData = collect_data(collectedData, tt, cols, satParameters, smallFragments, counter)
        counter += 1
        print()
    return collectedData


def main():
    '''
    Main function.

    Returns:
        None.
    '''
    start = time.time()
    simulationData = hubald_model(10000, 1200, 3)
    print(f'Number of collisions: {int(simulationData[2][-1])}')
    finish = time.time()
    print(f'Process finished after: {round(finish - start, 2)}s')

    plot_data(simulationData)


if __name__ == '__main__':
    main()
