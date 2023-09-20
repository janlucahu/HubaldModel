from calculations import *


def small_fragment(colProbMatrix, satParameters, satConstants, smallFragments, mm, bb, timestep, sigma, accuracy):
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
    probThresh = 10 ** (-10)
    satellitesStruck = 0
    activeSatParameters = satParameters[:, -1] == 1
    activeSatellites = np.count_nonzero(activeSatParameters)
    for _ in range(activeSatellites):
        pp = np.random.rand()
        fragmentCollisionProb = linear_distribution(smallFragments, mm, bb)
        if pp < fragmentCollisionProb:
            struckSat, act = np.where(satParameters == 1)
            randSat = struckSat[np.random.randint(0, len(struckSat))]
            satParameters[randSat][6] = 0
            print(f'Satellite {randSat} struck by fragment')
            satellitesStruck += 1

            mask = np.any(colProbMatrix == randSat, axis=1)
            colProbMatrix = colProbMatrix[~mask]

            for sat2 in range(satParameters.shape[0]):
                if not sat2 == randSat:
                    colProb = collision_probability(randSat, sat2, satParameters, satConstants, sigma, timestep,
                                                    accuracy)
                    if colProb > probThresh:
                        newRow = np.array([randSat, sat2, colProb])
                        colProbMatrix = np.vstack((colProbMatrix, newRow))

    return colProbMatrix, satParameters, satellitesStruck


def large_fragment(colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, mm, bb):
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
    for _ in range(len(nonDestroyed)):
        pp = np.random.rand()
        fragmentCollisionProb = linear_distribution(largeFragments, mm, bb)
        if pp < fragmentCollisionProb:
            randSat = nonDestroyed[np.random.randint(0, len(nonDestroyed))]
            satParameters[randSat][:] = np.array([0, 0, 0, 0, 0, 0, -1])
            satConstants[randSat][:] = np.array([0, 0, 0, 0, 0, 0])

            mask = np.any(colProbMatrix == randSat, axis=1)
            colProbMatrix = colProbMatrix[~mask]

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
    collisionsInIteration = 0
    shift = 0
    for ii in range(colProbMatrix.shape[0]):
        pp = np.random.rand()
        if pp < colProbMatrix[ii - shift][-1]:
            collisionsInIteration += 1
            sat1 = int(colProbMatrix[ii - shift][0])
            sat2 = int(colProbMatrix[ii - shift][1])
            freeIndices.append(sat1)
            freeIndices.append(sat2)

            print(
                '*****************************************************************************************************')
            print(f'Collision between satellites {sat2} and {sat1},   '
                  f'status: {satParameters[sat1][6]} {satParameters[sat2][6]},    '
                  f'iteration {tt} of {tmax}')
            print(
                '*****************************************************************************************************')

            satParameters[sat1][:] = np.array([0, 0, 0, 0, 0, 0, -1])
            satParameters[sat2][:] = np.array([0, 0, 0, 0, 0, 0, -1])
            satConstants[sat1][:] = np.array([0, 0, 0, 0, 0, 0])
            satConstants[sat2][:] = np.array([0, 0, 0, 0, 0, 0])

            mask = np.any(colProbMatrix == sat1, axis=1) | np.any(colProbMatrix == sat2, axis=1)
            for _, val in enumerate(mask):
                if val:
                    shift += 1
            colProbMatrix = colProbMatrix[~mask]

    fragments = (smallFragments, largeFragments)
    return colProbMatrix, satParameters, satConstants, fragments, collisionsInIteration, freeIndices


def deorbit_and_launch(colProbMatrix, satParameters, satConstants, aLimits, timestep, sigma, accuracy, freeIndices,
                       startsPerTimestep, deorbitsPerTimestep):
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
    probThresh = 10 ** (-10)
    deorbitingSats = np.random.randint(0, deorbitsPerTimestep)
    oldSats = []
    for oldSat in range(len(satParameters)):
        if satParameters[oldSat][6] == 0:
            oldSats.append(oldSat)
    if len(oldSats) > deorbitingSats:
        for _ in range(deorbitingSats):
            randIndex = np.random.randint(0, len(oldSats))
            randOldSat = oldSats[randIndex]
            del oldSats[randIndex]
            print(f'Deorbitation of satellite {randOldSat}')
            satParameters[randOldSat][:] = np.array([0, 0, 0, 0, 0, 0, -1])
            freeIndices.append(randOldSat)

            mask = np.any(colProbMatrix == randOldSat, axis=1)
            colProbMatrix = colProbMatrix[~mask]

    launchedSats = np.random.randint(0, startsPerTimestep)
    newPars, newCons = initialize(launchedSats, aLimits, 1, plane=False)
    launchIndices = []
    for ii in range(launchedSats):
        if len(freeIndices) > 0:
            newSat = freeIndices[0]
            del freeIndices[0]
            launchIndices.append(newSat)
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
            for sat2 in range(satParameters.shape[0]):
                if not sat2 == newSat:
                    colProb = collision_probability(newSat, sat2, satParameters, satConstants, sigma, timestep,
                                                    accuracy)
                    if colProb > probThresh:
                        newRow = np.array([newSat, sat2, colProb])
                        colProbMatrix = np.vstack((colProbMatrix, newRow))

        else:
            currentSatNr = satParameters.shape[0]
            launchIndices.append(currentSatNr)
            satParameters = np.vstack((satParameters, newPars[ii]))
            satConstants = np.vstack((satConstants, newCons[ii]))

            for sat2 in range(satParameters.shape[0] - 1):
                colProb = collision_probability(currentSatNr, sat2, satParameters, satConstants, sigma, timestep,
                                                accuracy)
                if colProb > probThresh:
                    newRow = np.array([currentSatNr, sat2, colProb])
                    colProbMatrix = np.vstack((colProbMatrix, newRow))

    if len(launchIndices) > 0:
        print(f'Launch of new satellites: {launchIndices}')

    return colProbMatrix, satParameters, satConstants, freeIndices
