import time
from calc import initialize, calculate_trig
from prob_matrix import build_prob_matrix, build_prob_matrix_parallel, build_prob_matrix_pool

num_sats = int(10000)
earth_radius = float(6_370_000)
a_low = float(earth_radius + 200_000)
a_high = float(earth_radius + 2_000_000)
active_fraction = float(0.5)
plane = bool(False)

sat_parameters, sat_constants = initialize(num_sats, a_low, a_high, active_fraction, plane)

accuracy = int(20)
sin = calculate_trig(accuracy, "s")
cos = calculate_trig(accuracy, "c")

sigma = float(10000)
time_step = int(3)
prob_thresh = float(10 ** (-15))
lower_bound =int(0)
upper_bound = int(sat_parameters.shape[0])
num_workers = int(12)

if __name__ == '__main__':
    start = time.time()
    single = build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                               lower_bound, upper_bound)
    finish = time.time()
    print(f"Single finished after {finish - start}s")

    start = time.time()
    njit = build_prob_matrix_parallel(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                      num_workers)
    finish = time.time()
    print(f"njit finished after {finish - start}s")

    start = time.time()
    pool = build_prob_matrix_pool(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                  num_workers)
    finish = time.time()
    print(f"pool finished after {finish - start}s")

    print(single.shape[0], njit.shape[0], pool.shape[0])
