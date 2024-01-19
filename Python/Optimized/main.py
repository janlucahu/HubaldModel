import time
from numba import jit
from calc import initialize, calculate_trig
from prob_matrix import build_prob_matrix, build_prob_matrix_parallel, build_prob_matrix_pool


@jit(nopython=True)
def single() -> None:
    num_sats = int(1000)
    earth_radius = float(6_370_000)
    a_low = float(earth_radius + 200_000)
    a_high = float(earth_radius + 2_000_000)
    active_fraction = float(0.5)
    plane = bool(False)

    sat_parameters, sat_constants = initialize(num_sats, a_low, a_high, active_fraction, plane)

    accuracy = int(20)
    sin = calculate_trig(accuracy, "s")
    cos = calculate_trig(accuracy, "c")

    sigma = float(2000)
    time_step = int(3)
    prob_thresh = float(10 ** (-15))
    lower_bound =int(0)
    upper_bound = int(sat_parameters.shape[0])
    prob_matrix = build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                    lower_bound, upper_bound)
    print(prob_matrix)

    return None


@jit(nopython=True)
def multi_jit():
    num_sats = int(1000)
    earth_radius = float(6_370_000)
    a_low = float(earth_radius + 200_000)
    a_high = float(earth_radius + 2_000_000)
    active_fraction = float(0.5)
    plane = bool(False)

    sat_parameters, sat_constants = initialize(num_sats, a_low, a_high, active_fraction, plane)

    accuracy = int(20)
    sin = calculate_trig(accuracy, "s")
    cos = calculate_trig(accuracy, "c")

    sigma = float(2000)
    time_step = int(3)
    prob_thresh = float(10 ** (-15))
    num_workers = 8
    prob_matrix = build_prob_matrix_parallel(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                               num_workers)
    print(prob_matrix)
    return None


def multi_pool():
    num_sats = int(1000)
    earth_radius = float(6_370_000)
    a_low = float(earth_radius + 200_000)
    a_high = float(earth_radius + 2_000_000)
    active_fraction = float(0.5)
    plane = bool(False)

    sat_parameters, sat_constants = initialize(num_sats, a_low, a_high, active_fraction, plane)

    accuracy = int(20)
    sin = calculate_trig(accuracy, "s")
    cos = calculate_trig(accuracy, "c")

    sigma = float(2000)
    time_step = int(3)
    prob_thresh = float(10 ** (-15))
    num_workers = 8
    prob_matrix = build_prob_matrix_pool(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                         num_workers)
    print(prob_matrix)

    return None


if __name__ == '__main__':
    start = time.time()
    single()
    finish = time.time()
    print(f"Single finished after {finish - start}s.")

    start = time.time()
    multi_jit()
    finish = time.time()
    print(f"Multi jit finished after {finish - start}s.")

    start = time.time()
    multi_pool()
    finish = time.time()
    print(f"Multi Pool finished after {finish - start}s.")
