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
    satIndices = []
    for ii in range(satParameters.shape[0]):
        satIndices.append(ii)
    calculationSlices = calculation_slices(satIndices, numberOfWorkers)
    for ii in range(len(calculationSlices) - 1):
        if ii == 0:
            workerCalculations = calculationSlices[ii][-1] ** 2 / 2 - calculationSlices[ii][-1]
        else:
            workerCalculations = (calculationSlices[ii][-1] ** 2 / 2 - calculationSlices[ii][-1] / 2 -
                                  (calculationSlices[ii -1][-1] ** 2 / 2 - calculationSlices[ii - 1][-1] / 2))
        print(f"Worker {ii + 1}: Calculations: {workerCalculations}")
    with Pool() as pool:
        processes = []
        for ii in range(len(calculationSlices) - 1):
            lower = calculationSlices[ii]
            upper = calculationSlices[ii + 1]
            p = pool.apply_async(sparse_prob_matrix, args=[satParameters, satConstants, sigma, timestep, lower, upper,
                                                           ii, acc])
            processes.append(p)

        for process in processes:
            process.get()


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
    main2()
    finish = time.time()
    print(f"Finalized after {finish - start}s")
