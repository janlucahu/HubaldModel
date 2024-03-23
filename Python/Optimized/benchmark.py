import os
import csv
import time
import numpy as np
from calc import initialize, calculate_trig
from prob_matrix import build_prob_matrix, build_prob_matrix_parallel, build_prob_matrix_pool


def benchmark(num_workers_list, matrix_size_list, mode, output_dir):
    sigma = 2000
    time_step = 3
    accuracy = 20
    prob_thresh = 10 ** (-15)
    sin = calculate_trig(accuracy, 's')
    cos = calculate_trig(accuracy, 'c')
    file_path = os.path.join(output_dir, "benchmark.csv")
    for num_workers in num_workers_list:
        for matrix_size in matrix_size_list:
            print("Now running: ")
            print(f"Number of workers: {num_workers}")
            print(f"Matrix size: {matrix_size}")
            print(f"Mode: {mode}")
            sat_parameters, sat_constants = initialize(matrix_size, 200_000, 2_000_000, 0.5, False)
            if num_workers == 1:
                start = time.time()
                build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos, 0,
                                  sat_parameters.shape[0])
            elif mode == "njit":
                start = time.time()
                build_prob_matrix_parallel(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin,
                                           cos, num_workers)
            elif mode == "pool":
                start = time.time()
                build_prob_matrix_pool(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                       num_workers)
            finish = time.time()
            elapsed_time = finish - start
            print(f"Finished after {elapsed_time}s")
            print()
            stats = [mode, num_workers, matrix_size, elapsed_time]
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(stats)


if __name__ == '__main__':
    num_workers_list = [1, 4, 8, 12]
    matrix_size_list = [30000, 40000, 50000]
    modes = ["pool", "njit"]
    output_dir = os.path.join(os.getcwd(), "output")
    for mode in modes:
        benchmark(num_workers_list, matrix_size_list, mode, output_dir)
