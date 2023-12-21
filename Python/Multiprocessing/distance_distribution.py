import os
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from scipy.stats import pareto
from scipy.optimize import curve_fit
from multiprocessing import Pool
from calculations import initialize
from split_calculations import find_minimum, calculation_slices, collision_probability


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


@jit(nopython=True)
def col_matrix_single(nrOfSats):
    sigma = 2000
    timestep = 3
    acc = 20
    satParameters, satConstants = initialize(nrOfSats, [200_000 + 6_370_000, 2_000_000 + 6_370_000], 0.5)
    numSats = satParameters.shape[0]
    numDistances = int(numSats ** 2 / 2 - numSats / 2)
    probabilities = np.empty(numDistances)

    ind = 0
    for sat1 in range(numSats):
        print(f"{sat1} of {numSats}")
        for sat2 in range(sat1):
            colProb = collision_probability(sat1, sat2, satParameters, satConstants, sigma, timestep, acc)
            probabilities[ind] = colProb
            ind += 1

    return probabilities


# Function to process each chunk and write non-zero elements to a file
def process_chunk(chunk):
    # Identify zero elements and mark their indices
    zero_indices = chunk.index[chunk.iloc[:, 0] == 0]

    # Remove zero elements from the chunk
    non_zero_chunk = chunk.drop(zero_indices)

    # Write non-zero elements to a CSV file
    non_zero_chunk.to_csv(output_file, mode='a', index=False, header=False)


# Define the exponential decay function
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


# Define the chunk size and file paths
chunk_size = 1_000_000  # Adjust the chunk size as per your requirements
input_file = "/Users/janlucal/Documents/GitHub/HubaldModel/Python/Multiprocessing/Input/Matrices/probabilities/20000.csv"
output_file = "/Users/janlucal/Documents/GitHub/HubaldModel/Python/Multiprocessing/Input/Matrices/probabilities/20000_reduced.csv"


# Read the input CSV file in chunks
reader = pd.read_csv(input_file, header=None, chunksize=chunk_size)
for chunk in reader:
    # Process the chunk
    process_chunk(chunk)

"""
print("Importing data")
data = np.genfromtxt("/Users/janlucal/Documents/GitHub/HubaldModel/Python/Multiprocessing/Input/Matrices/probabilities/50000_reduced.csv")
print("Import succesful")
prob = np.sort(data)[::-1]
xx = np.linspace(0, prob.shape[0], prob.shape[0])
non_zero = np.nonzero(prob)

# Generate example data
x = np.linspace(0, prob.shape[0], prob.shape[0])
y = prob

# Fit the exponential decay function to the data
popt, pcov = curve_fit(exponential_decay, x, y)

# Get the optimized parameters
print(popt)
a_opt, b_opt, c_opt = popt

# Generate the fitted curve
y_fit = exponential_decay(x, a_opt, b_opt, c_opt)

# Plot the original data and the fitted curve
plt.scatter(x, y, label='Data')
plt.plot(x, y_fit, label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Exclude zero and negative values
data = data[data > 0]

# Create logarithmically spaced bins
min_value = data.min()
max_value = data.max()
epsilon = 0  # Small positive offset to avoid zero values
bin_edges = np.logspace(np.log10(min_value + epsilon), np.log10(max_value + epsilon), num=30)

# Compute the histogram
hist, _ = np.histogram(data, bins=bin_edges)

# Calculate the bin centers
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

# Plot the double logarithmic histogram
plt.plot(bin_centers, hist, 'bo', label='Data')
plt.xscale("log")

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
"""
"""
sats = [50000]
probabilities = []
norm = True
for ii, satNr in enumerate(sats):
    probability = col_matrix_single(satNr)
    fileName = "probabilities.csv"
    currentDir = os.getcwd()
    outputDir = os.path.join(currentDir, os.path.abspath("output"))
    fileDir = os.path.join(outputDir, fileName)
    np.savetxt(fileDir, probability, delimiter=',')
    probabilities.append(probability)
    bins = range(0, int(np.max(probability)), 10000)

    dis, bins = np.histogram(probability, bins)
    if norm:
        dis = dis / np.sum(probability)

    plt.plot(bins[0:-1], dis, label=f'{satNr}')

plt.legend()
plt.plot()



start = time.time()
distance = np.genfromtxt("/Users/janlucal/Documents/GitHub/HubaldModel/Python/Multiprocessing/Input/Matrices/distances/20000.csv")
end = time.time()
print(f"Array imported after {round(end - start, 2)}s")

stat_distances = np.empty(20000)
start = time.time()
for ii in range(stat_distances.shape[0]):
    ind = np.random.randint(0, distance.shape[0])
    stat_distances[ii] = distance[ind]
end = time.time()
print(f"Statistical distances built after {round(end - start, 2)}")

bins = range(0, int(np.max(stat_distances)), 10000)
dis, bins = np.histogram(stat_distances, bins)
dis = dis / np.sum(stat_distances)
plt.plot(bins[0:-1], dis, label='statistical distances')

bins = range(0, int(np.max(distance)), 10000)
dis, bins = np.histogram(distance, bins)
dis = dis / np.sum(distance)
plt.plot(bins[0:-1], dis, label='distance distribution')

start = time.time()
normalized_dist = distance / np.sum(distance)
cdf = np.cumsum(normalized_dist)
end = time.time()
print(f"Statistical preparation finished after {round(end - start, 2)}")

stat_distances2 = np.empty(20000)
start = time.time()
for ii in range(stat_distances2.shape[0]):
    random_num = np.random.uniform()
    sample_index = np.searchsorted(cdf, random_num)
    sampled_distance = distance[sample_index]
    stat_distances2[ii] = sampled_distance
end = time.time()
print(f"Second statistical distances built after {round(end - start, 2)}")

bins = range(0, int(np.max(stat_distances2)), 10000)
dis, bins = np.histogram(stat_distances2, bins)
dis = dis / np.sum(stat_distances2)
plt.plot(bins[0:-1], dis, label='statistical distances 2')

plt.legend()
plt.show()
"""
