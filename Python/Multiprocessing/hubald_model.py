import os
import sys
import time
import multiprocessing
import numpy as np
from model_dynamics import (small_fragment, large_fragment, satellite_collision, deorbit_and_launch, small_stat,
                            deorbit_launch_stat)
from data_handling import collect_data
from calculations import initialize
from split_calculations import (build_prob_matrix, calculation_slices, build_dis_matrix, calculation_slices2,
                                build_stat_prob_matrix)
from file_io import save_arrays, read_arrays


def hubald_model(input_parameters, saveDir, reuseArrays="", accuracy=20):
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
    earthRadius = 6_370_000
    startingSats = input_parameters.get("starting_sats")
    sigma = input_parameters.get("sigma")
    fragmentColProb = input_parameters.get("fragment_collision_prob")
    activePercentage = input_parameters.get("active_percentage")
    aLimits = input_parameters.get("aLimits")
    aLimits[0] += earthRadius
    aLimits[1] += earthRadius
    smallFragments = input_parameters.get("small_fragments")
    largeFragments = input_parameters.get("large_fragments")
    timestep = input_parameters.get("timestep")
    tmax = input_parameters.get("tmax")
    startsPerTimestep = input_parameters.get("starts_per_timestep")
    deorbitsPerTimestep = input_parameters.get("deorbits_per_timestep")
    collectedData = np.empty((10, tmax // timestep))
    freeIndices = []

    availableCores = multiprocessing.cpu_count()
    numWorkers = availableCores

    if reuseArrays:
        satParameters = read_arrays(os.path.join(reuseArrays, "satParameters.csv"))
        satConstants = read_arrays(os.path.join(reuseArrays, "satConstants.csv"))
        colProbMatrix = read_arrays(os.path.join(reuseArrays, "probabilityMatrix.csv"))
        print(f"Successfully imported array of {satParameters.shape[0]} satellites")
    else:
        satParameters, satConstants = initialize(startingSats, aLimits, activePercentage, plane=False)
        print("Number of CPU cores:", availableCores)
        satIndices = []
        for ii in range(0, satParameters.shape[0], 1):
            satIndices.append(ii)
        calculationSlices = calculation_slices(satIndices, satParameters, numWorkers)
        totalCalculations = int(satParameters.shape[0] ** 2 / 2 - satParameters.shape[0] / 2)

        print("Calculating probability matrix")
        print(f"Total calculations: {totalCalculations}")
        saveDistances = True
        if saveDistances and not reuseArrays:
            distanceArray = build_stat_prob_matrix(satParameters, satConstants, numWorkers, timestep, sigma, accuracy)
            arraysList = [satParameters, satConstants, distanceArray]
            save_arrays(arraysList, saveDir)
            sys.exit("Distance array has been created.")

        start = time.time()
        colProbMatrix = build_prob_matrix(calculationSlices, satParameters, satConstants, sigma, timestep, accuracy)
        finish = time.time()
        print(f"Matrix built after {round(finish - start), 2}s")

    arraysList = [satParameters, satConstants, colProbMatrix]
    save_arrays(arraysList, saveDir)

    counter = 0
    for tt in range(0, tmax, timestep):
        print(f"Iteration {tt} of {tmax}")
        # m, b = 1 / 10000 / 12 / 100000000 * timestep, 0
        m, b = fragmentColProb / 12 / 100000000 * timestep, 0
        colProbMatrix, satParameters, satsStruck = small_fragment(colProbMatrix, satParameters, satConstants,
                                                                  smallFragments, m, b, timestep, sigma, accuracy)
        smallFragmentCols = satsStruck

        # m, b = 1 / 10000 / 12 / 100000 * timestep, 0
        m, b = fragmentColProb / 12 / 100000 * timestep, 0
        fragmentArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, m, b)
        colProbMatrix, satParameters, satConstants, fragments, satsStruck, freeIndices = large_fragment(*fragmentArgs)
        largeFragmentsCols = satsStruck
        smallFragments, largeFragments = fragments

        colArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, tt, tmax)
        colProbMatrix, satParameters, satConstants, fragments, cols, freeIndices = satellite_collision(*colArgs)
        smallFragments, largeFragments = fragments

        colProbMatrix, satParameters, satConstants, freeIndices = deorbit_and_launch(colProbMatrix, satParameters,
                                                                                     satConstants, aLimits, timestep,
                                                                                     sigma, accuracy, freeIndices,
                                                                                     startsPerTimestep,
                                                                                     deorbitsPerTimestep, numWorkers)
        nonZeroRows = satParameters[:, 0] != 0
        numberOfSatellites = np.count_nonzero(nonZeroRows)
        print(f'Number of satellites: {numberOfSatellites}    Iteration: {tt}')
        collectedData = collect_data(collectedData, tt, cols, satParameters, smallFragments, largeFragments,
                                     smallFragmentCols, largeFragmentsCols, counter)
        counter += 1
        print()
    return collectedData, colProbMatrix


def hubald_model_statistical(input_parameters, saveDir, reuseArrays="", accuracy=20):
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
    earthRadius = 6_370_000
    startingSats = input_parameters.get("starting_sats")
    sigma = input_parameters.get("sigma")
    fragmentColProb = input_parameters.get("fragment_collision_prob")
    activePercentage = input_parameters.get("active_percentage")
    aLimits = input_parameters.get("aLimits")
    aLimits[0] += earthRadius
    aLimits[1] += earthRadius
    smallFragments = input_parameters.get("small_fragments")
    largeFragments = input_parameters.get("large_fragments")
    timestep = input_parameters.get("timestep")
    tmax = input_parameters.get("tmax")
    startsPerTimestep = input_parameters.get("starts_per_timestep")
    deorbitsPerTimestep = input_parameters.get("deorbits_per_timestep")
    collectedData = np.empty((10, tmax // timestep))
    freeIndices = []

    print("Importing distance matrix...")
    distancesFile = os.path.abspath("/Users/janlucal/Documents/GitHub/HubaldModel/Python/Multiprocessing/Input/Matrices/distances/1000.csv")
    distances = np.genfromtxt(distancesFile, delimiter=',')
    print(f"Distance matrix imported containing {distances.shape[0]} values")

    availableCores = multiprocessing.cpu_count()
    numWorkers = availableCores

    if reuseArrays:
        satParameters = read_arrays(os.path.join(reuseArrays, "satParameters.csv"))
        satConstants = read_arrays(os.path.join(reuseArrays, "satConstants.csv"))
        colProbMatrix = read_arrays(os.path.join(reuseArrays, "probabilityMatrix.csv"))
        print(f"Successfully imported array of {satParameters.shape[0]} satellites")
    else:
        satParameters, satConstants = initialize(startingSats, aLimits, activePercentage, plane=False)
        print("Number of CPU cores:", availableCores)
        satIndices = []
        for ii in range(0, satParameters.shape[0], 1):
            satIndices.append(ii)
        calculationSlices = calculation_slices2(satIndices, satParameters, numWorkers)
        print(calculationSlices)
        totalCalculations = int(satParameters.shape[0] ** 2 / 2 - satParameters.shape[0] / 2)

        print("Calculating probability matrix")
        print(f"Total calculations: {totalCalculations}")
        saveDistances = False
        if saveDistances and not reuseArrays:
            distanceArray = build_dis_matrix(satParameters, satConstants, numWorkers, accuracy)
            arraysList = [satParameters, satConstants, distanceArray]
            save_arrays(arraysList, saveDir)
            sys.exit("Distance array has been created.")

        start = time.time()
        colProbMatrix = build_prob_matrix(calculationSlices, satParameters, satConstants, sigma, timestep, accuracy)
        finish = time.time()
        print(f"Matrix built after {round(finish - start), 2}s")

    arraysList = [satParameters, satConstants, colProbMatrix]
    save_arrays(arraysList, saveDir)

    counter = 0
    for tt in range(0, tmax, timestep):
        print(f"Iteration {tt} of {tmax}")
        # m, b = 1 / 10000 / 12 / 100000000 * timestep, 0
        m, b = fragmentColProb / 12 / 100000000 * timestep, 0
        colProbMatrix, satParameters, satsStruck = small_stat(colProbMatrix, satParameters, smallFragments, m, b,
                                                              timestep, sigma, distances)
        smallFragmentCols = satsStruck

        # m, b = 1 / 10000 / 12 / 100000 * timestep, 0
        m, b = fragmentColProb / 12 / 100000 * timestep, 0
        fragmentArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, m, b)
        colProbMatrix, satParameters, satConstants, fragments, satsStruck, freeIndices = large_fragment(*fragmentArgs)
        largeFragmentsCols = satsStruck
        smallFragments, largeFragments = fragments

        colArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, tt, tmax)
        colProbMatrix, satParameters, satConstants, fragments, cols, freeIndices = satellite_collision(*colArgs)
        smallFragments, largeFragments = fragments

        colProbMatrix, satParameters, satConstants, freeIndices = deorbit_launch_stat(colProbMatrix, satParameters,
                                                                                      satConstants, aLimits, timestep,
                                                                                      sigma, freeIndices,
                                                                                      startsPerTimestep,
                                                                                      deorbitsPerTimestep, distances)
        nonZeroRows = satParameters[:, 0] != 0
        numberOfSatellites = np.count_nonzero(nonZeroRows)
        print(f'Number of satellites: {numberOfSatellites}    Iteration: {tt}')
        collectedData = collect_data(collectedData, tt, cols, satParameters, smallFragments, largeFragments,
                                     smallFragmentCols, largeFragmentsCols, counter)
        counter += 1
        print()
    return collectedData, colProbMatrix
