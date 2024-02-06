from calc import initialize, calculate_trig
from prob_matrix import build_prob_matrix, update_prob_matrix

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
col_prob_matrix = build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                    lower_bound, upper_bound)

launched_sats = int(10)

col_prob_matrix = update_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                     col_prob_matrix, launched_sats, a_low, a_high, active_fraction, plane)

print(col_prob_matrix)
