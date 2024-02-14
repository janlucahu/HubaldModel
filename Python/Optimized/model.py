import os
import time
import shutil
import numpy as np
from numba import jit
from numba import types
from data_handling import read_csv, plot_data
from calc import initialize, calculate_trig
from prob_matrix import build_prob_matrix, build_prob_matrix_parallel
from dynamics import small_fragment, large_fragment, satellite_collision, deorbits, starts
from file_io import read_input_file, create_header, write_results_to_csv, save_arrays


@jit(nopython=True)
def kessler_model(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
                  col_prob_matrix: np.ndarray[np.float64, 2], num_sats: int, sigma: float, frag_col_prob: float,
                  active_fraction: float, a_low: float, a_high: float, num_small_fragments: int,
                  num_large_fragments: int, starts_per_timestep: int, deorbits_per_timestep: int, time_step: int,
                  tmax: int, mode: str, num_workers: int, launch_mode: str) -> np.ndarray[np.float64, 2]:

    plane = False
    earth_radius = float(6_370_000)
    a_low += earth_radius
    a_high += earth_radius

    accuracy = int(20)
    sin = calculate_trig(accuracy, "s")
    cos = calculate_trig(accuracy, "c")
    prob_thresh = float(10 ** (-15))
    lower_bound = int(0)
    upper_bound = int(sat_parameters.shape[0])

    initial_small_fragments = num_small_fragments
    initial_large_fragments = num_large_fragments

    if mode == "calculate":
    	col_prob_matrix = build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh,
                                        sin, cos, lower_bound, upper_bound)

    simulation_data = np.empty((10, tmax // time_step))
    counter = 0

    for tt in range(0, tmax, time_step):
        print(f"Timestep {tt} of {tmax}\n")
        # small fragment collisions
        mm_small = frag_col_prob / 12 / initial_small_fragments * time_step
        small_args = (col_prob_matrix, sat_parameters, sat_constants, num_small_fragments, mm_small, time_step, sigma,
                      accuracy, prob_thresh, sin, cos)
        col_prob_matrix, sat_parameters, small_frag_collisions = small_fragment(*small_args)

        # large fragment collisions
        mm_large = frag_col_prob / 12 / initial_large_fragments * time_step
        large_args = (col_prob_matrix, sat_parameters, sat_constants, num_small_fragments, num_large_fragments,
                      mm_large)
        col_prob_matrix, sat_parameters, sat_constants, fragments, large_frag_collisions = large_fragment(*large_args)
        num_small_fragments, num_large_fragments = fragments

        # satellite collisions
        sat_col_args = (col_prob_matrix, sat_parameters, sat_constants, num_small_fragments, num_large_fragments)
        col_prob_matrix, sat_parameters, sat_constants, fragments, sat_collisions = satellite_collision(*sat_col_args)
        num_small_fragments, num_large_fragments = fragments

        # satellite deorbits
        deorbit_args = (col_prob_matrix, sat_parameters, sat_constants, deorbits_per_timestep)
        col_prob_matrix, sat_parameters, sat_constants = deorbits(*deorbit_args)

        # satellite starts1
        start_args = (col_prob_matrix, sat_parameters, sat_constants, num_workers, sigma, time_step, accuracy,
                      prob_thresh, sin, cos, starts_per_timestep, a_low, a_high, active_fraction, plane, launch_mode)
        col_prob_matrix, sat_parameters, sat_constants = starts(*start_args)

        inactive_sats = np.where(sat_parameters[:, -1] == 0)[0].shape[0]
        active_sats = np.where(sat_parameters[:, -1] == 1)[0].shape[0]
        total_sats = active_sats + inactive_sats

        simulation_data[0][counter] = tt
        simulation_data[1][counter] = sat_collisions
        simulation_data[2][counter] = np.sum(simulation_data[1, 0:counter])
        simulation_data[3][counter] = total_sats
        simulation_data[4][counter] = active_sats
        simulation_data[5][counter] = inactive_sats
        simulation_data[6][counter] = num_small_fragments
        simulation_data[7][counter] = num_large_fragments
        simulation_data[8][counter] = small_frag_collisions
        simulation_data[9][counter] = large_frag_collisions

    return simulation_data, sat_parameters, sat_constants, col_prob_matrix


def simulation(input_file):
    print("Simulation started with following input parameters:")
    print(f"Input path:\n{input_file}\n")
    start = time.time()
    starting_time = time.asctime()
    current_dir = os.getcwd()
    current_time = time.strftime("%a %b %d %H:%M:%S %Y")  # Get the current time in the desired format
    current_time = current_time.replace(" ", "_")  # Replace spaces with underscores
    current_time = current_time[4:]
    current_time = current_time.replace(":", "-")  # Replace colons with hyphens or any other desired character
    save_dir = os.path.join(current_dir, os.path.abspath("output/" + current_time))
    os.makedirs(save_dir)
    destination_file = os.path.join(save_dir, os.path.basename(input_file))
    shutil.copy(input_file, destination_file)

    input_parameters = read_input_file(input_file)
    matrix_mode = input_parameters.get("matrix_mode")
    launch_mode = input_parameters.get("launch_mode")
    accuracy = input_parameters.get("accuracy")
    prob_thresh = float(eval(input_parameters.get("prob_thresh")))
    num_sats = input_parameters.get("starting_sats")
    sigma = input_parameters.get("sigma")
    frag_col_prob = input_parameters.get("fragment_collision_prob")
    active_fraction = input_parameters.get("active_percentage")
    earth_radius = 6_370_000
    a_limits = input_parameters.get("aLimits")
    a_low = a_limits[0] + earth_radius
    a_high = a_limits[1] + earth_radius
    num_small_fragments = input_parameters.get("small_fragments")
    num_large_fragments = input_parameters.get("large_fragments")
    starts_per_timestep = input_parameters.get("starts_per_timestep")
    deorbits_per_timestep = input_parameters.get("deorbits_per_timestep")
    time_step = input_parameters.get("timestep")
    tmax = input_parameters.get("tmax")
    num_workers = input_parameters.get("num_workers")
    for key, val in input_parameters.items():
        print(f"{key} = {val}")

    if matrix_mode == "calculate":
        print("Calculating starting matrix.")
        sin = calculate_trig(accuracy, 's')
        cos = calculate_trig(accuracy, 'c')
        sat_parameters, sat_constants = initialize(num_sats, a_low, a_high, active_fraction, False)
        if launch_mode == "single":
            lower_bound = int(0)
            upper_bound = sat_parameters.shape[0]
            col_prob_matrix = build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh,
                                                sin, cos, lower_bound, upper_bound)
        elif launch_mode == "njit":
            col_prob_matrix = build_prob_matrix_parallel(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh,
                                                         sin, cos, num_workers)
        else:
            raise ValueError("Invalid initial launch mode. Check Input file.")

    elif matrix_mode == "import":
        print("Importing data.")
        path = r"/Users/janlucal/Documents/GitHub/HubaldModel/Python/Optimized/input/Matrices/50000/1"
        sat_parameters = read_csv(path + r"/satParameters.csv")
        sat_constants = read_csv(path + r"/satConstants.csv")
        col_prob_matrix = read_csv(path + r"/probabilityMatrix.csv")
    else:
        raise ValueError("Invalid initial matrix mode. Check Input file.")
    start_arrays = {"sat_parameters": sat_parameters, "sat_constants": sat_constants, "prob_matrix": col_prob_matrix}
    save_arrays(start_arrays, save_dir)

    print("Kessler model employed.\n")
    model_args = (sat_parameters, sat_constants, col_prob_matrix, num_sats, sigma, frag_col_prob, active_fraction,
                  a_low, a_high, num_small_fragments, num_large_fragments, starts_per_timestep, deorbits_per_timestep,
                  time_step, tmax, matrix_mode, num_workers, launch_mode)
    sim_data, sat_parameters, sat_constants, col_prob_matrix = kessler_model(*model_args)
    finish = time.time()
    elapsed_time = finish - start
    ending_time = time.asctime()
    print(f'Simulation finished after: {round(elapsed_time, 2)}s')
    print("Writing results.")
    plot_data(sim_data, save_dir, input_parameters)
    time_stamps = [starting_time, ending_time, elapsed_time]
    file_header = create_header(time_stamps, input_parameters)
    write_results_to_csv(sim_data, file_header, save_dir)
    end_arrays = {"sat_parameters": sat_parameters, "sat_constants": sat_constants, "prob_matrix": col_prob_matrix}
    save_arrays(end_arrays, save_dir, end=True)
    print("Process finished.\n\n")


if __name__ == "__main__":
    input_path = os.path.join(os.getcwd(), "input/input_parameters.txt")
    simulation(input_path)
