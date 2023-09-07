from model_dynamics_sparse import *
from data_handling import *


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
    sigma = 10000
    activePercentage = 0.3
    smallFragments = 100_000_000
    largeFragments = 100_000
    collectedData = np.empty((7, tmax // timestep))
    freeIndices = []

    satParameters, satConstants = initialize(startingSats, aLimits, activePercentage, plane=False)
    colProbMatrix = sparse_prob_matrix(satParameters, satConstants, sigma, timestep, accuracy)
    counter = 0
    for tt in range(0, tmax, timestep):
        m, b = 1 / 10000 / 12 / 100000000 * timestep, 0
        colProbMatrix, satParameters, satsStruck = small_fragment(colProbMatrix, satParameters, satConstants,
                                                                  smallFragments, m, b, timestep, sigma, accuracy)

        m, b = 1 / 10000 / 12 / 100000 * timestep, 0
        fragmentArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, m, b)
        colProbMatrix, satParameters, satConstants, fragments, satsStruck, freeIndices = large_fragment(*fragmentArgs)
        smallFragments, largeFragments = fragments

        colArgs = (colProbMatrix, satParameters, satConstants, smallFragments, largeFragments, freeIndices, tt, tmax)
        colProbMatrix, satParameters, satConstants, fragments, cols, freeIndices = satellite_collision(*colArgs)
        smallFragments, largeFragments = fragments

        colProbMatrix, satParameters, satConstants, freeIndices = deorbit_and_launch(colProbMatrix, satParameters,
                                                                                     satConstants, aLimits, timestep,
                                                                                     sigma, accuracy, freeIndices)
        nonZeroRows = satParameters[:, 0] != 0
        numberOfSatellites = np.count_nonzero(nonZeroRows)
        print(f'Number of satellites: {numberOfSatellites}    Iteration: {tt}')
        collectedData = collect_data(collectedData, tt, cols, satParameters, smallFragments, counter)
        counter += 1
        print()
    return collectedData, colProbMatrix