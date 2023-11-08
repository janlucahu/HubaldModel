import numpy as np
import os
import time
import matplotlib.pyplot as plt


def collect_data(collectedData, tt, collisions, satParameters, smallFragments, largeFragments, smallFragmentCols,
                 largeFragmentCols, counter):
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
    collectedData[6][counter] = smallFragments
    collectedData[7][counter] = largeFragments
    collectedData[8][counter] = smallFragmentCols
    collectedData[9][counter] = largeFragmentCols

    return collectedData


def plot_data(simulationData, saveDir):
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
    smallFragments = simulationData[6]
    largeFragments = simulationData[7]
    totalFragments = smallFragments + largeFragments
    smallFragCols = simulationData[8]
    largeFragCols = simulationData[9]

    fig, axs = plt.subplots(3, 2, figsize=(12, 8))

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

    axs[1, 1].plot(tt, smallFragments, label="small")
    axs[1, 1].plot(tt, largeFragments, label="large")
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Number of fragments')
    axs[1, 1].set_title('Fragments over time')
    axs[1, 1].legend()


    axs[2, 0].plot(tt, smallFragCols, label="small")
    axs[2, 0].plot(tt, largeFragCols, label="large")
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("Fragment collisions per timestep")
    axs[2, 0].set_title("Fragment collisions per iteration")
    axs[2, 0].legend()

    axs[2, 1].plot(totalFragments, totalSatellites)
    axs[2, 1].set_xlabel("Total number of fragments")
    axs[2, 1].set_ylabel("Total number of satellites")
    axs[2, 1].set_title("Total satellites as function of total fragments")

    plt.tight_layout()
    saveDir = os.path.join(saveDir, "hubald_simulation.png")
    plt.savefig(saveDir, dpi=600)
