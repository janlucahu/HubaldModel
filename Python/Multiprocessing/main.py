import time
import multiprocessing
import numpy as np
from calculations import initialize
from collision_probability import calc_collision_probability
from split_calculations import calculation_slices, sparse_prob_matrix
from multiprocessing import Pool


def main():
    aLimits = [200_000, 2_000_000]
    activeFraction = 0.3
    satParameters, satConstants = initialize(500, aLimits, activeFraction)
    sigma = 2000
    timestep = 3
    acc = 20

    E_1 = np.linspace(0, 2 * np.pi, acc)
    E_2 = np.linspace(0, 2 * np.pi, acc)

    E1, E2 = np.meshgrid(E_1, E_2)
    sinE, cosE = np.sin(E1), np.cos(E1)

    with Pool() as pool:
        processes = []
        for sat1 in range(satParameters.shape[0]):
            for sat2 in range(sat1):
                p = pool.apply_async(calc_collision_probability, args=[satParameters[sat1], satParameters[sat2],
                                                                        satConstants[sat1], satConstants[sat2],
                                                                        sigma, timestep, sinE, cosE])
                processes.append(p)

        for process in processes:
            process.get()


def main2():
    print("Test parallel calculation")
    aLimits = [200_000, 2_000_000]
    activeFraction = 0.3
    satParameters, satConstants = initialize(2000, aLimits, activeFraction)
    sigma = 2000
    timestep = 3
    acc = 20

    availableCores = multiprocessing.cpu_count()
    print("Number of CPU cores:", availableCores)
    numberOfWorkers = availableCores
    satIndices = [20, 77, 564]
    for ii in range(800, satParameters.shape[0], 1):
        satIndices.append(ii)
    calculationSlices = calculation_slices(satIndices, numberOfWorkers)
    results = []
    with Pool() as pool:
        processes = []
        for sliceIndices in calculationSlices:
            p = pool.apply_async(sparse_prob_matrix, args=[satParameters, satConstants, sigma, timestep,
                                                           sliceIndices, acc])
            processes.append(p)

        for process in processes:
            result = process.get()  # Get the result from each process
            results.append(result)  # Store the result in the results list

        # Stack the results into one big array
        probMatrix = np.concatenate(results, axis=0)

        return probMatrix


def test_non_parallel():
    print("Test non parallel calculation")
    aLimits = [200_000, 2_000_000]
    activeFraction = 0.3
    satParameters, satConstants = initialize(5000, aLimits, activeFraction)
    sigma = 2000
    timestep = 3
    acc = 20

    probMatrix = sparse_prob_matrix(satParameters, satConstants, sigma, timestep, 0, satParameters.shape[0], 0, acc)


if __name__ == '__main__':
    start = time.time()
    probMatrix = main2()
    print(probMatrix)
    finish = time.time()
    print(f"Finalized after {finish - start}s")
