import time
from calc import initialize, calculate_trig
from prob_matrix import build_prob_matrix, update_prob_matrix, parallel_update
from dynamics import small_fragment, large_fragment, satellite_collision, deorbits, starts

num_sats = int(10)
earth_radius = float(6_370_000)
a_low = float(earth_radius + 200_000)
a_high = float(earth_radius + 2_000_000)
active_fraction = float(0.5)
plane = bool(False)

sat_parameters, sat_constants = initialize(num_sats, a_low, a_high, active_fraction, plane)

accuracy = int(20)
sin = calculate_trig(accuracy, "s")
cos = calculate_trig(accuracy, "c")

sigma = float(5000000)
time_step = int(3)
prob_thresh = float(10 ** (-15))
lower_bound =int(0)
upper_bound = int(sat_parameters.shape[0])
col_prob_matrix = build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                    lower_bound, upper_bound)

frag_col_prob = 4
num_small_fragments = 100000000
mm = frag_col_prob / 12 / num_small_fragments * time_step
small_fragment_args = (col_prob_matrix, sat_parameters, sat_constants, num_small_fragments, mm,
                       time_step, sigma, accuracy, prob_thresh, sin, cos)

# print(sat_parameters[:, 6])
# start = time.time()
# col_prob_matrix, sat_parameters, satellites_struck = small_fragment(*small_fragment_args)
# finish = time.time()
# print(sat_parameters[:, 6])
# print(f"Matrix updated after {finish - start}s")

frag_col_prob = 4
num_large_fragments = 1000000
mm = frag_col_prob / 12 / num_large_fragments * time_step
large_fragment_args = (col_prob_matrix, sat_parameters, sat_constants, num_small_fragments, num_large_fragments, mm)

# print(sat_parameters)
# col_prob_matrix, sat_parameters, sat_constants, _, _ = large_fragment(*large_fragment_args)
# print(sat_parameters)

# print(col_prob_matrix)
# sat_collision_args = (col_prob_matrix, sat_parameters, sat_constants, num_small_fragments, num_large_fragments)
# col_prob_matrix, sat_parameters, sat_constants, fragments, num_collisions = satellite_collision(*sat_collision_args)
# print(col_prob_matrix)

deorbits_per_timestep = int(10)
deorbit_args = (col_prob_matrix, sat_parameters, sat_constants, deorbits_per_timestep)

# print(sat_parameters[:, 6])
# col_prob_matrix, sat_parameters, sat_constants = deorbits(*deorbit_args)
# print(sat_parameters[:, 6])

mode = "single"
num_workers = 8
launched_sats = int(5)

launch = {"mode": mode, "num_workers": num_workers, "sigma": sigma, "time_step": time_step, "accuracy": accuracy,
          "prob_thresh": prob_thresh, "sin": sin, "cos": cos, "starts_per_timestep": launched_sats, "a_low": a_low,
          "a_high": a_high, "active_fraction": active_fraction, "plane": plane}

print(sat_parameters.shape[0])
col_prob_matrix, sat_parameters, sat_constants = starts(col_prob_matrix, sat_parameters, sat_constants, launch)
print(sat_parameters.shape[0])
