import numpy as np
from probability_distributions import linear_distribution
from calculations import initialize, collision_probability
from split_calculations import calculation_slices, sparse_prob_matrix, build_prob_matrix
from multiprocessing import Pool


def small_fragment(colProbMatrix, satParameters, satConstants, smallFragments, mm, bb, timestep, sigma, accuracy):
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
            #print(f'Satellite {randSat} struck by fragment')
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
            #print(f'Satellite {randSat} struck by large fragment')
            satellitesStruck += 1
            freeIndices.append(randSat)
    fragments = (smallFragments, largeFragments)
    return colProbMatrix, satParameters, satConstants, fragments, satellitesStruck, freeIndices


def satellite_collision(colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices,
                        tt, tmax):
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

            #print(
            #    '*****************************************************************************************************')
            #print(f'Collision between satellites {sat2} and {sat1},   '
            #      f'status: {satParameters[sat1][6]} {satParameters[sat2][6]},    '
            #      f'iteration {tt} of {tmax}')
            #print(
            #    '*****************************************************************************************************')

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
                       startsPerTimestep, deorbitsPerTimestep, numberOfWorkers):

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
            #print(f'Deorbitation of satellite {randOldSat}')
            satParameters[randOldSat][:] = np.array([0, 0, 0, 0, 0, 0, -1])
            freeIndices.append(randOldSat)

            mask = np.any(colProbMatrix == randOldSat, axis=1)
            colProbMatrix = colProbMatrix[~mask]

    launchedSats = np.random.randint(0, startsPerTimestep)
    newPars, newCons = initialize(launchedSats, aLimits, 1, plane=False)
    launchIndices = []
    satIndices = []

    if len(freeIndices) > 0 and len(freeIndices) >= launchedSats:
        satIndices = freeIndices[0:launchedSats]
        for ii in range(launchedSats):
            newSat = freeIndices[ii]
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
        del freeIndices[0:launchedSats]
        calculationSlices = calculation_slices(satIndices, numberOfWorkers)
        launchIndices = satIndices

        probMatrix = build_prob_matrix(calculationSlices, satParameters, satConstants, sigma, timestep,
                                       accuracy)
        colProbMatrix = np.vstack((colProbMatrix, probMatrix))

    elif len(freeIndices) > 0 and len(freeIndices) < launchedSats:
        satIndices = freeIndices
        for ii, newSat in enumerate(freeIndices):
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
        for ii in range(launchedSats - len(freeIndices)):
            currentSatNr = satParameters.shape[0]
            index = launchedSats - len(freeIndices) + ii
            satParameters = np.vstack((satParameters, newPars[index]))
            satConstants = np.vstack((satConstants, newCons[index]))
            satIndices.append(currentSatNr)
        del freeIndices[0:-1]
        calculationSlices = calculation_slices(satIndices, numberOfWorkers)
        launchIndices = satIndices

        probMatrix = build_prob_matrix(calculationSlices, satParameters, satConstants, sigma, timestep,
                                       accuracy)
        colProbMatrix = np.vstack((colProbMatrix, probMatrix))

    else:
        for ii in range(launchedSats):
            currentSatNr = satParameters.shape[0]
            launchIndices.append(currentSatNr)
            satParameters = np.vstack((satParameters, newPars[ii]))
            satConstants = np.vstack((satConstants, newCons[ii]))
            satIndices.append(currentSatNr)
            calculation_slices(satIndices, numberOfWorkers)
            launchIndices = satIndices

            probMatrix = build_prob_matrix(calculationSlices, satParameters, satConstants, sigma, timestep,
                                           accuracy)
            colProbMatrix = np.vstack((colProbMatrix, probMatrix))

    #if len(launchIndices) > 0:
        #print(f'Launch of new satellites: {launchIndices}')

    return colProbMatrix, satParameters, satConstants, freeIndices
