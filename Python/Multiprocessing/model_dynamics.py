import numpy as np
from probability_distributions import linear_distribution
from calculations import initialize
from split_calculations import indice_slices, build_prob_matrix2, collision_probability, col_prob_stat, col_prob_stat2
from distance_distribution import exponential_decay

b = 261000


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
        pp = np.random.uniform(0, 1)
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

    deorbitingSats = deorbitsPerTimestep
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

    launchedSats = startsPerTimestep
    newPars, newCons = initialize(launchedSats, aLimits, 1, plane=False)
    launchIndices = []
    satIndices = []

    print(f"Sats launched: {launchedSats}")

    if len(freeIndices) > 0 and len(freeIndices) >= launchedSats:
        satIndices = freeIndices[0:launchedSats]
        for ii in range(launchedSats):
            newSat = freeIndices[ii]
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
        del freeIndices[0:launchedSats]
        calculationSlices = indice_slices(satIndices, satParameters, numberOfWorkers)
        launchIndices = satIndices

        probMatrix = build_prob_matrix2(calculationSlices, satParameters, satConstants, sigma, timestep, accuracy)
        colProbMatrix = np.vstack((colProbMatrix, probMatrix))

    elif 0 < len(freeIndices) < launchedSats:
        satIndices = []
        index = 0
        for ii, newSat in enumerate(freeIndices):
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
            satIndices.append(newSat)
        for ii in range(launchedSats - len(freeIndices)):
            currentSatNr = satParameters.shape[0]
            index = len(freeIndices) + ii
            satParameters = np.vstack((satParameters, newPars[index]))
            satConstants = np.vstack((satConstants, newCons[index]))
            satIndices.append(currentSatNr)
        del freeIndices[0:-1]
        calculationSlices = indice_slices(satIndices, satParameters, numberOfWorkers)
        launchIndices = satIndices

        probMatrix = build_prob_matrix2(calculationSlices, satParameters, satConstants, sigma, timestep, accuracy)
        colProbMatrix = np.vstack((colProbMatrix, probMatrix))

    else:
        for ii in range(launchedSats):
            currentSatNr = satParameters.shape[0]
            launchIndices.append(currentSatNr)
            satParameters = np.vstack((satParameters, newPars[ii]))
            satConstants = np.vstack((satConstants, newCons[ii]))
            satIndices.append(currentSatNr)

        launchIndices = satIndices
        calculationSlices = indice_slices(satIndices, satParameters, numberOfWorkers)
        probMatrix = build_prob_matrix2(calculationSlices, satParameters, satConstants, sigma, timestep, accuracy)
        colProbMatrix = np.vstack((colProbMatrix, probMatrix))

    if len(launchIndices) > 0:
        print(f'Launch of new satellites: {launchIndices}')

    return colProbMatrix, satParameters, satConstants, freeIndices


def small_stat(colProbMatrix, satParameters, smallFragments, mm, bb, timestep, sigma):
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
                    randNum = np.random.rand()
                    colProb = exponential_decay(randNum, 1, b, 0)
                    if colProb > probThresh:
                        newRow = np.array([randSat, sat2, colProb])
                        colProbMatrix = np.vstack((colProbMatrix, newRow))

    return colProbMatrix, satParameters, satellitesStruck


def deorbit_launch_stat(colProbMatrix, satParameters, satConstants, aLimits, timestep, sigma, freeIndices,
                        startsPerTimestep, deorbitsPerTimestep):

    colProbThresh = 10 ** (-10)

    deorbitingSats = deorbitsPerTimestep
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

    launchedSats = startsPerTimestep
    newPars, newCons = initialize(launchedSats, aLimits, 1, plane=False)
    launchIndices = []
    satIndices = []

    print(f"Sats launched: {launchedSats}")
    print(freeIndices)

    if len(freeIndices) > 0 and len(freeIndices) >= launchedSats:
        satIndices = freeIndices[0:launchedSats]
        del freeIndices[0:launchedSats]
        satIndices = np.sort(satIndices, kind='quicksort')[::-1]
        usedIndice = []
        for ii, newSat in enumerate(satIndices):
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
            for sat2 in range(satParameters.shape[0]):
                if sat2 not in usedIndice and sat2 != newSat:
                    randNum = np.random.rand()
                    colProb = exponential_decay(randNum, 1, b, 0)
                    if colProb > colProbThresh:
                        newRow = np.array([newSat, sat2, colProb])
                        colProbMatrix = np.vstack((colProbMatrix, newRow))
                    usedIndice.append(newSat)

    elif 0 < len(freeIndices) < launchedSats:
        for ii, newSat in enumerate(freeIndices):
            satIndices.append(newSat)
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
        currentSatNr = satParameters.shape[0]
        for ii in range(launchedSats - len(freeIndices)):
            index = currentSatNr + ii
            satIndices.append(index)
            satParameters = np.vstack((satParameters, newPars[ii + len(freeIndices)]))
            satConstants = np.vstack((satConstants, newCons[ii + len(freeIndices)]))
        del freeIndices[0:-1]

        satIndices = np.sort(satIndices, kind='quicksort')[::-1]
        usedIndice = []
        for ii, newSat in enumerate(satIndices):
            print(f"{ii} of {len(satIndices)}")
            for sat2 in range(satParameters.shape[0]):
                if sat2 not in usedIndice and sat2 != newSat:
                    randNum = np.random.rand()
                    colProb = exponential_decay(randNum, 1, b, 0)
                    if colProb > colProbThresh:
                        newRow = np.array([newSat, sat2, colProb])
                        colProbMatrix = np.vstack((colProbMatrix, newRow))
                    usedIndice.append(newSat)

    else:
        for ii in range(launchedSats):
            currentSatNr = satParameters.shape[0]
            launchIndices.append(currentSatNr)
            satParameters = np.vstack((satParameters, newPars[ii]))
            satConstants = np.vstack((satConstants, newCons[ii]))
            satIndices.append(currentSatNr)

        satIndices = np.sort(satIndices, kind='quicksort')[::-1]
        usedIndice = []
        for ii, newSat in enumerate(satIndices):
            satParameters[newSat] = newPars[ii]
            satConstants[newSat] = newCons[ii]
            for sat2 in range(satParameters.shape[0]):
                if sat2 not in usedIndice and sat2 != newSat:
                    randNum = np.random.rand()
                    colProb = exponential_decay(randNum, 1, b, 0)
                    if colProb > colProbThresh:
                        newRow = np.array([newSat, sat2, colProb])
                        colProbMatrix = np.vstack((colProbMatrix, newRow))
                    usedIndice.append(newSat)

    if len(launchIndices) > 0:
        print(f'Launch of new satellites: {launchIndices}')

    return colProbMatrix, satParameters, satConstants, freeIndices
