import os
import numpy as np
import matplotlib.pyplot as plt


def read_csv(filename):
    # Use numpy to read the CSV file into a regular numpy array
    data = np.genfromtxt(filename, delimiter=',')

    return data


def plot_data(simulationData, saveDir, input_parameters):
    '''
    Plots the gathered simulation data.
    Args:
        simulationData (2darray): Measured data.

    Returns:
        None.
    '''
    startingSats = input_parameters.get("starting_sats")
    sigma = input_parameters.get("sigma")
    fragmentColProb = input_parameters.get("fragment_collision_prob")
    activePercentage = input_parameters.get("active_percentage")
    smallStartFragments = input_parameters.get("small_fragments")
    largeStartFragments = input_parameters.get("large_fragments")
    timestep = input_parameters.get("timestep")
    tmax = input_parameters.get("tmax")
    startsPerTimestep = input_parameters.get("starts_per_timestep")
    deorbitsPerTimestep = input_parameters.get("deorbits_per_timestep")

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

    axs[1, 1].plot(tt, smallFragments)
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Number of fragments')
    axs[1, 1].set_title('Small fragments over time')

    axs[2, 0].plot(tt, smallFragCols, label="small")
    axs[2, 0].plot(tt, largeFragCols, label="large")
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("Fragment collisions per timestep")
    axs[2, 0].set_title("Fragment collisions per iteration")
    axs[2, 0].legend()

    axs[2, 1].plot(tt, largeFragments)
    axs[2, 1].set_xlabel('Time')
    axs[2, 1].set_ylabel('Number of fragments')
    axs[2, 1].set_title('Large fragments over time')

    fig.suptitle(f"starting satellites: {startingSats},    starts: {startsPerTimestep},    deorbits: {deorbitsPerTimestep}\nactive percentage: {activePercentage},    sigma: {sigma},    fragment collision p: {fragmentColProb}\nsmall fragments: {smallStartFragments},    large fragments: {largeStartFragments}")

    plt.tight_layout()
    saveDir = os.path.join(saveDir, "hubald_simulation.png")
    plt.savefig(saveDir, dpi=600)
