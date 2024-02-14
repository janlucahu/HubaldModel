import numpy as np
from numba import jit
from numba import types
from calc import collision_probability
from prob_matrix import update_prob_matrix, parallel_update, pool_update


@jit(nopython=True)
def small_fragment(col_prob_matrix: np.ndarray[np.float64, 2], sat_parameters: np.ndarray[np.float64, 2],
                   sat_constants: np.ndarray[np.float64, 2], num_small_fragments: float, mm: float, time_step: int,
                   sigma: float, accuracy: int, prob_thresh: float, sin: np.ndarray[np.float64, 2],
                   cos: np.ndarray[np.float64, 2]) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2], int]:

    satellites_struck = int(0)
    active_indices = np.where(sat_parameters[:, 6] == 1)[0]
    for _ in range(active_indices.shape[0]):
        pp = np.random.rand()
        frag_col_prob = mm * num_small_fragments
        if pp < frag_col_prob:
            rand_ind = np.random.randint(0, len(active_indices))
            struck_sat = active_indices[rand_ind]

            removed_index = np.empty(active_indices.shape[0] - 1, dtype=types.int64)
            removed_index[:rand_ind] = active_indices[:rand_ind]
            removed_index[rand_ind:] = active_indices[rand_ind + 1:]
            active_indices = removed_index

            sat_parameters[struck_sat][6] = 0
            satellites_struck += 1

            struck_sat_in_matrix = np.where(col_prob_matrix == struck_sat)[0]

            for ii in range(struck_sat_in_matrix.shape[0]):
                ind = struck_sat_in_matrix[ii]
                sat1 = int(col_prob_matrix[ind, 0])
                sat2 = int(col_prob_matrix[ind, 1])
                col_prob = collision_probability(sat_parameters, sat_constants, sat1, sat2, sigma, time_step, accuracy,
                                                 sin, cos)
                if col_prob > prob_thresh:
                    col_prob_matrix[ind, 0] = sat1
                    col_prob_matrix[ind, 1] = sat2
                    col_prob_matrix[ind, 2] = col_prob
                else:
                    removed_index = np.empty((col_prob_matrix.shape[0] - 1, col_prob_matrix.shape[1]),
                                             dtype=types.float64)

                    removed_index[:ind] = col_prob_matrix[:ind]
                    removed_index[ind:] = col_prob_matrix[ind + 1:]
                    col_prob_matrix = removed_index

    return col_prob_matrix, sat_parameters, satellites_struck


@jit(nopython=True)
def large_fragment(col_prob_matrix: np.ndarray[np.float64, 2], sat_parameters: np.ndarray[np.float64, 2],
                   sat_constants: np.ndarray[np.float64, 2], num_small_fragments: float, num_large_fragments: float,
                   mm) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2], np.ndarray[np.float64, 2],
                                tuple[float, float], int]:

    non_destroyed_indices = np.where(sat_parameters[:, 6] != -1)[0]
    satellites_struck = int(0)
    for ii in range(non_destroyed_indices.shape[0]):
        pp = np.random.rand()
        frag_col_prob = mm * num_large_fragments
        if pp < frag_col_prob:
            destroyed_sat = int(non_destroyed_indices[ii])
            sat_parameters[destroyed_sat][:] = np.array([0, 0, 0, 0, 0, 0, -1], dtype=types.float64)
            sat_constants[destroyed_sat][:] = np.array([0, 0, 0, 0, 0, 0], dtype=types.float64)

            struck_sat_in_matrix = np.where(col_prob_matrix == destroyed_sat)[0]
            shift = int(0)
            for ii in range(struck_sat_in_matrix.shape[0]):
                ind = struck_sat_in_matrix[ii] - shift
                removed_index = np.empty((col_prob_matrix.shape[0] - 1, col_prob_matrix.shape[1]),
                                          dtype=types.float64)
                removed_index[:ind] = col_prob_matrix[:ind]
                removed_index[ind:] = col_prob_matrix[ind + 1:]
                col_prob_matrix = removed_index
                shift += 1

            num_small_fragments += 9_000
            num_large_fragments += 200
            satellites_struck += 1

    fragments = (num_small_fragments, num_large_fragments)
    return col_prob_matrix, sat_parameters, sat_constants, fragments, satellites_struck


@jit(nopython=True)
def satellite_collision(col_prob_matrix: np.ndarray[np.float64, 2], sat_parameters: np.ndarray[np.float64, 2],
                        sat_constants: np.ndarray[np.float64, 2], num_small_fragments: float,
                        num_large_fragments: float) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2],
                                                             np.ndarray[np.float64, 2], tuple[float, float], int]:

    num_collisions = int(0)
    shift = int(0)
    for ii in range(col_prob_matrix.shape[0]):
        if ii - shift < col_prob_matrix.shape[0]:
            pp = np.random.uniform(0, 1)
            if pp < col_prob_matrix[ii - shift][2] or col_prob_matrix[ii - shift][2] >= 1:
                num_collisions += 1
                num_small_fragments += 2 * 9_000
                num_large_fragments += 2 * 200
                sat1 = int(col_prob_matrix[ii - shift][0])
                sat2 = int(col_prob_matrix[ii - shift][1])

                sat_parameters[sat1][:] = np.array([0, 0, 0, 0, 0, 0, -1], dtype=types.float64)
                sat_parameters[sat2][:] = np.array([0, 0, 0, 0, 0, 0, -1], dtype=types.float64)
                sat_constants[sat1][:] = np.array([0, 0, 0, 0, 0, 0], dtype=types.float64)
                sat_constants[sat2][:] = np.array([0, 0, 0, 0, 0, 0], dtype=types.float64)

                struck_sat_in_matrix = np.where(col_prob_matrix[:, 0:2] == sat1)[0]
                s1 = int(0)
                for jj in range(struck_sat_in_matrix.shape[0]):
                    ind = struck_sat_in_matrix[jj] - s1
                    removed_index = np.empty((col_prob_matrix.shape[0] - 1, col_prob_matrix.shape[1]),
                                              dtype=types.float64)
                    removed_index[:ind] = col_prob_matrix[:ind]
                    removed_index[ind:] = col_prob_matrix[ind + 1:]
                    col_prob_matrix = removed_index
                    s1 += 1

                struck_sat_in_matrix = np.where(col_prob_matrix[:, 0:2] == sat2)[0]
                s2 = int(0)
                for jj in range(struck_sat_in_matrix.shape[0]):
                    ind = struck_sat_in_matrix[jj] - s2
                    removed_index = np.empty((col_prob_matrix.shape[0] - 1, col_prob_matrix.shape[1]),
                                              dtype=types.float64)
                    removed_index[:ind] = col_prob_matrix[:ind]
                    removed_index[ind:] = col_prob_matrix[ind + 1:]
                    col_prob_matrix = removed_index
                    s2 += 1
                shift += 1

        else:
            break

    fragments = (num_small_fragments, num_large_fragments)
    return col_prob_matrix, sat_parameters, sat_constants, fragments, num_collisions


@jit(nopython=True)
def deorbits(col_prob_matrix: np.ndarray[np.float64, 2], sat_parameters: np.ndarray[np.float64, 2],
             sat_constants: np.ndarray[np.float64, 2], deorbits_per_timestep: int) -> tuple[np.ndarray[np.float64, 2],
                                                                                            np.ndarray[np.float64, 2],
                                                                                            np.ndarray[np.float64, 2]]:

    inactive_satellites = np.where(sat_parameters[:, 6] == 0)[0]
    if inactive_satellites.shape[0] < deorbits_per_timestep:
        deorbits_per_timestep = inactive_satellites.shape[0]
    deorbiting_indices = np.empty(deorbits_per_timestep, dtype=types.int64)
    for ii in range(deorbits_per_timestep):
        rand_ind = np.random.randint(0, inactive_satellites.shape[0])
        deorbiting_indices[ii] = int(inactive_satellites[rand_ind])
        new_inactive = np.empty(inactive_satellites.shape[0] - 1, dtype=types.int64)
        new_inactive[:rand_ind] = inactive_satellites[:rand_ind]
        new_inactive[rand_ind:] = inactive_satellites[rand_ind + 1:]
        inactive_satellites = new_inactive

    for ii in range(deorbiting_indices.shape[0]):
        deorbiting_sat = int(deorbiting_indices[ii])
        sat_parameters[deorbiting_sat][:] = np.array([0, 0, 0, 0, 0, 0, -1], dtype=types.float64)
        sat_constants[deorbiting_sat][:] = np.array([0, 0, 0, 0, 0, 0], dtype=types.float64)

        struck_sat_in_matrix = np.where(col_prob_matrix[:, 0:2] == deorbiting_sat)[0]
        s1 = int(0)
        for jj in range(struck_sat_in_matrix.shape[0]):
            ind = struck_sat_in_matrix[jj] - s1
            removed_index = np.empty((col_prob_matrix.shape[0] - 1, col_prob_matrix.shape[1]),
                                      dtype=types.float64)
            removed_index[:ind] = col_prob_matrix[:ind]
            removed_index[ind:] = col_prob_matrix[ind + 1:]
            col_prob_matrix = removed_index
            s1 += 1

    return col_prob_matrix, sat_parameters, sat_constants


@jit(nopython=True)
def starts(col_prob_matrix: np.ndarray[np.float64, 2], sat_parameters: np.ndarray[np.float64, 2],
           sat_constants: np.ndarray[np.float64, 2], num_workers: int, sigma: float, time_step: int, accuracy: int,
           prob_thresh: float, sin: np.ndarray[np.float64, 2], cos: np.ndarray[np.float64, 2], starts_per_timestep: int,
           a_low: float, a_high: float, active_fraction: float, plane: bool,
           mode: str) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2], np.ndarray[np.float64, 2]]:

    launched_sats = int(starts_per_timestep)
    if mode == "single":
        col_prob_matrix, sat_parameters, sat_constants = update_prob_matrix(sat_parameters, sat_constants, sigma,
                                                                            time_step, accuracy, prob_thresh, sin, cos,
                                                                            col_prob_matrix, launched_sats, a_low,
                                                                            a_high, active_fraction, plane)
    elif mode == "njit":
        col_prob_matrix, sat_parameters, sat_constants = parallel_update(sat_parameters, sat_constants, sigma,
                                                                         time_step, accuracy, prob_thresh, sin, cos,
                                                                         col_prob_matrix, launched_sats, a_low, a_high,
                                                                         active_fraction, plane, num_workers)
    # elif mode == "pool":
    #     col_prob_matrix, sat_parameters, sat_constants = pool_update(sat_parameters, sat_constants, sigma,
    #                                                                  time_step, accuracy, prob_thresh, sin, cos,
    #                                                                  col_prob_matrix, launched_sats, a_low, a_high,
    #                                                                  active_fraction, plane, num_workers)
    else:
        raise ValueError("Unsupported collision probability update mode.")

    return col_prob_matrix, sat_parameters, sat_constants
