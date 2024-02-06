import time
from numba import jit
from calc import initialize, calculate_trig, collision_probability


@jit(nopython=True)
def main() -> None:
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

    for sat1 in range(sat_parameters.shape[0]):
        for sat2 in range(sat1):
            col_prob = collision_probability(sat_parameters, sat_constants, sat1, sat2, sigma, time_step, accuracy,
                                             sin, cos)
            if col_prob > 0:
                print(col_prob)

    return None


if __name__ == '__main__':
    start = time.time()
    main()
    finish = time.time()
    print(f"Process finished after {finish - start}s.")
