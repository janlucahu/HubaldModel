import time
import multiprocessing
import numpy as np
from calculations import initialize
from collision_probability import calc_collision_probability
from split_calculations import calculation_slices, sparse_prob_matrix, build_prob_matrix, indice_slices
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


def main2(satParameters, satConstants, sigma, timestep, acc):
    print("Test parallel calculation")

    availableCores = multiprocessing.cpu_count()
    print("Number of CPU cores:", availableCores)
    numberOfWorkers = availableCores
    satIndices = []
    for ii in range(0, satParameters.shape[0], 1):
        satIndices.append(ii)
    calculationSlices = calculation_slices(satIndices, numberOfWorkers)

    probMatrix = build_prob_matrix(calculationSlices, satParameters, satConstants, sigma, timestep, acc)

    return probMatrix


def test_non_parallel(satParameters, satConstants, sigma, timestep, acc):
    print("Test non parallel calculation")

    satIndices = []
    for ii in range(0, satParameters.shape[0], 1):
        satIndices.append(ii)
    probMatrix = sparse_prob_matrix(satParameters, satConstants, sigma, timestep, satIndices, acc)

    return probMatrix


def benchmark_computation(startingSats):
    aLimits = [200_000, 2_000_000]
    activeFraction = 0.3
    sigma = 2000
    timestep = 3
    acc = 20

    single = []
    multi = []
    singleArr = []
    multiArr = []
    for sats in startingSats:
        satParameters, satConstants = initialize(sats, aLimits, activeFraction)

        start = time.time()
        probMatrix = test_non_parallel(satParameters, satConstants, sigma, timestep, acc)
        print(probMatrix.shape[0])
        finish = time.time()
        elapsedTime = finish - start
        print(f"Finalized after {elapsedTime}s")
        single.append(elapsedTime)
        singleArr.append(probMatrix.shape[0])
        print("Single-core computation:")
        print(single)
        print(singleArr)

        start = time.time()
        probMatrix = main2(satParameters, satConstants, sigma, timestep, acc)
        print(probMatrix.shape[0])
        finish = time.time()
        elapsedTime = finish - start
        print(f"Finalized after {elapsedTime}s")
        multi.append(elapsedTime)
        multiArr.append(probMatrix.shape[0])
        print("Multi-core computation:")
        print(multi)
        print(multiArr)

    return single, singleArr, multi, multiArr


if __name__ == '__main__':
    indices = [3, 5, 8]
    satParameters, satConstants = initialize(20, [200_000, 2_000_000], 0.3)
    for ii in range(15, satParameters.shape[0], 1):
        indices.append(ii)

    slices = indice_slices(indices, satParameters, 10)
    for ii in range(len(slices)):
        print(len(slices[ii]))
    print(slices)
