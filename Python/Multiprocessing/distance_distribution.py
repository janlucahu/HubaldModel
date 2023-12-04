import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from calculations import initialize
from split_calculations import find_minimum, calculation_slices


def part_dis_matrix(satParameters, satConstants, satIndices, acc):
    distList = []
    for ii, sat1 in enumerate(satIndices):
        for sat2 in range(satParameters.shape[0]):
            if sat2 not in satIndices[0:ii] and sat1 != sat2:
                parameters1 = satParameters[sat1]
                parameters2 = satParameters[sat2]
                const1 = satConstants[sat1]
                const2 = satConstants[sat2]
                minDistance = find_minimum(parameters1, parameters2, const1, const2, acc)

                distList.extend([sat1, sat2, minDistance])

    partDisMatrix = np.array(distList).reshape(-1, 3)

    return partDisMatrix


def build_distance_matrix(nrOfSats):
    acc=20
    satParameters, satConstants = initialize(nrOfSats, [200_000 + 6_370_000, 2_000_000 + 6_370_000], 0.5)

    numWorkers = multiprocessing.cpu_count()
    satIndices = []
    for ii in range(0, satParameters.shape[0], 1):
        satIndices.append(ii)
    calculationSlices = calculation_slices(satIndices, satParameters, numWorkers)

    results = []
    with Pool() as pool:
        processes = []
        for sliceIndices in calculationSlices:
            p = pool.apply_async(part_dis_matrix, args=[satParameters, satConstants, sliceIndices, acc])
            processes.append(p)

        for process in processes:
            result = process.get()  # Get the result from each process
            results.append(result)  # Store the result in the results list

        # Stack the results into one big array
        distMatrix = np.concatenate(results, axis=0)

    return distMatrix


def dis_matrix_single(nrOfSats):
    acc=20
    satParameters, satConstants = initialize(nrOfSats, [200_000 + 6_370_000, 2_000_000 + 6_370_000], 0.5)
    numSats = satParameters.shape[0]
    numDistances = int(numSats ** 2 / 2 - numSats / 2)
    distances = np.empty(numDistances)

    ind = 0
    for sat1 in range(numSats):
        for sat2 in range(sat1):
            parameters1 = satParameters[sat1]
            parameters2 = satParameters[sat2]
            const1 = satConstants[sat1]
            const2 = satConstants[sat2]
            minDistance = find_minimum(parameters1, parameters2, const1, const2, acc)
            distances[ind] = minDistance
            ind += 1

    return distances

sats = [1000, 2000, 3000]
distances = []
norm = True
for ii, satNr in enumerate(sats):
    distance = dis_matrix_single(satNr)
    distances.append(distance)
    bins = range(0, int(np.max(distance)), 10000)

    dis, bins = np.histogram(distance, bins)
    if norm:
        dis = dis / np.sum(distance)
        bins = bins / np.sum(distance)

    plt.plot(bins[0:-1], dis, label=f'{satNr}')

plt.legend()
plt.plot()
