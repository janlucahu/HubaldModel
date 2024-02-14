import numpy as np
from numba import jit
from numba import types
from numba import njit, prange
from multiprocessing import Pool
from calc import collision_probability, initialize


@jit(nopython=True)
def stack_arrays(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    if len(arr2.shape) > 1:
        stacked = np.empty((arr1.shape[0] + arr2.shape[0], arr1.shape[1]), dtype=arr1.dtype)
    else:
        stacked = np.empty((arr1.shape[0] + 1, arr1.shape[1]), dtype=arr1.dtype)
    stacked[:arr1.shape[0], :] = arr1
    stacked[arr1.shape[0]:, :] = arr2
    return stacked


@jit(nopython=True)
def build_prob_matrix(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2], sigma: float,
                      time_step: int, accuracy: int, prob_thresh: float, sin: np.ndarray[np.float64, 2],
                      cos: np.ndarray[np.float64, 2], lower_bound: int, upper_bound: int) -> np.ndarray[np.float64, 2]:

    col_prob_matrix = np.empty((0, 3), dtype=types.float64)
    for sat1 in range(lower_bound, upper_bound, 1):
        for sat2 in range(sat1):
            col_prob = collision_probability(sat_parameters, sat_constants, sat1, sat2, sigma, time_step, accuracy, sin, cos)
            if col_prob > prob_thresh:
                new_row = np.empty(3, dtype=types.float64)
                new_row[0] = float(sat1)
                new_row[1] = float(sat2)
                new_row[2] = col_prob

                col_prob_matrix = stack_arrays(col_prob_matrix, new_row)

    return col_prob_matrix


@njit(parallel=True)
def build_prob_matrix_parallel(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
                               sigma: float, time_step: int, accuracy: int, prob_thresh: float,
                               sin: np.ndarray[np.float64, 2], cos: np.ndarray[np.float64, 2],
                               num_workers: int) -> np.ndarray[np.float64, 2]:

    num_calculations = sat_parameters.shape[0] ** 2 / 2 - sat_parameters.shape[0] / 2
    calc_per_worker = np.ceil(num_calculations / num_workers)
    calc_slices = np.empty(num_workers + 1, dtype=types.int64)
    calc_slices[0] = int(0)
    ind = 1
    low_bound = 0
    for sat in range(sat_parameters.shape[0]):
        if ind == num_workers:
            calc_slices[ind] = sat_parameters.shape[0] - 1
            break
        else:
            worker_calculations = sat ** 2 / 2 - sat / 2 - (low_bound ** 2 / 2 - low_bound / 2)
            if worker_calculations >= calc_per_worker:
                calc_slices[ind] = sat
                low_bound = calc_slices[ind]
                ind += 1

    results = []
    for ii in prange(num_workers):
        lower_bound = calc_slices[ii]
        upper_bound = calc_slices[ii + 1]
        result = build_prob_matrix(sat_parameters, sat_constants, sigma, time_step, accuracy, prob_thresh, sin, cos,
                                   lower_bound, upper_bound)
        results.append(result)

    col_prob_matrix = results[0]
    for matrix in results[1:]:
        col_prob_matrix = stack_arrays(col_prob_matrix, matrix)

    return col_prob_matrix


def build_prob_matrix_pool(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
                           sigma: float, time_step: int, accuracy: int, prob_thresh: float,
                           sin: np.ndarray[np.float64, 2], cos: np.ndarray[np.float64, 2],
                           num_workers: int) -> np.ndarray[np.float64, 2]:

    num_calculations = sat_parameters.shape[0] ** 2 / 2 - sat_parameters.shape[0] / 2
    calc_per_worker = np.ceil(num_calculations / num_workers)
    calc_slices = np.empty(num_workers + 1, dtype=np.int64)
    calc_slices[0] = int(0)
    ind = 1
    low_bound = 0
    for sat in range(sat_parameters.shape[0]):
        if ind == num_workers:
            calc_slices[ind] = sat_parameters.shape[0] - 1
            break
        else:
            worker_calculations = sat ** 2 / 2 - sat / 2 - (low_bound ** 2 / 2 - low_bound / 2)
            if worker_calculations >= calc_per_worker:
                calc_slices[ind] = sat
                low_bound = calc_slices[ind]
                ind += 1

    results = []
    with Pool() as pool:
        processes = []
        for ii in range(num_workers):
            lower_bound = calc_slices[ii]
            upper_bound = calc_slices[ii + 1]
            p = pool.apply_async(build_prob_matrix, args=[sat_parameters, sat_constants, sigma, time_step, accuracy,
                                                          prob_thresh, sin, cos, lower_bound, upper_bound])
            processes.append(p)

        for process in processes:
            result = process.get()
            results.append(result)

    col_prob_matrix = results[0]
    for matrix in results[1:]:
        col_prob_matrix = stack_arrays(col_prob_matrix, matrix)

    return col_prob_matrix


@jit(nopython=True)
def update(indices: list, sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
           sigma: float, time_step: int, accuracy: int, sin: np.ndarray[np.float64, 2], cos: np.ndarray[np.float64, 2],
           prob_thresh: float, worker: int, num_workers: int) -> np.ndarray[np.float64, 2]:

    new_col_probs = np.empty((0, 3), dtype=types.float64)
    indices_per_worker = int(np.ceil(len(indices / num_workers)))
    lower_bound = worker * indices_per_worker
    if worker != num_workers - 1:
        upper_bound = (worker +  1) * indices_per_worker
    else:
        upper_bound = len(indices)
    ind = int(1)
    for sat1 in indices[lower_bound:upper_bound]:
        for sat2 in range(sat_parameters.shape[0]):
            if sat2 not in indices[0:lower_bound + ind]: # to ensure newly launched satellites pair calculations are not redundant
                col_prob = collision_probability(sat_parameters, sat_constants, sat1, sat2, sigma, time_step, accuracy,
                                                 sin, cos)
                if col_prob > prob_thresh:
                    new_row = np.empty((1, 3), dtype=types.float64)
                    new_row[0][0] = float(sat1)
                    new_row[0][1] = float(sat2)
                    new_row[0][2] = col_prob

                    new_col_probs = stack_arrays(new_col_probs, new_row)

        ind += 1
    return new_col_probs


@jit(nopython=True)
def update_prob_matrix(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
                       sigma: float, time_step: int, accuracy: int, prob_thresh: float, sin: np.ndarray[np.float64, 2],
                       cos: np.ndarray[np.float64, 2], col_prob_matrix: np.ndarray[np.float64, 2],
                       launched_sats: int, a_low: float, a_high: float, active_fraction: float,
                       plane: bool) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2],
                                             np.ndarray[np.float64, 2]]:

    # check if reusable indices are sufficient or if new rows have to be appended to sat_parameters
    indices = np.where(sat_parameters[:, -1] == -1)[0]
    if launched_sats - len(indices) > 0:
        num_new_parameters = int(launched_sats - len(indices))
    else:
        num_new_parameters = int(0)

    new_indices = np.empty(num_new_parameters, dtype=np.int64)
    for ii in range(num_new_parameters):
        new_indices[ii] = sat_parameters.shape[0] + ii

    # indices of all newly launched satellites
    extended_indices = np.empty(indices.shape[0] + new_indices.shape[0], dtype=np.int64)
    extended_indices[:indices.shape[0]] = indices
    extended_indices[indices.shape[0]:] = new_indices

    # parameters of newly launched satellites
    new_parameters, new_constants = initialize(launched_sats, a_low, a_high, active_fraction, plane)
    for ii, ind in enumerate(indices):
        if ii > num_new_parameters:
            break
        sat_parameters[ind] = new_parameters[ii]
        sat_constants[ind] = new_constants[ii]

    sat_parameters = stack_arrays(sat_parameters, new_parameters[indices.shape[0]:])
    sat_constants = stack_arrays(sat_constants, new_constants[indices.shape[0]:])
    for ii, ind in enumerate(new_indices):
        sat_parameters[ind] = new_parameters[indices.shape[0] + ii]
        sat_constants[ind] = new_constants[indices.shape[0] + ii]

    indices = extended_indices

    worker = int(0)
    num_workers = int(1)
    new_col_probs = update(indices, sat_parameters, sat_constants, sigma, time_step, accuracy, sin, cos, prob_thresh,
                           worker, num_workers)
    col_prob_matrix = stack_arrays(col_prob_matrix, new_col_probs)

    return col_prob_matrix, sat_parameters, sat_constants


@njit(parallel=True)
def parallel_update(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
                    sigma: float, time_step: int, accuracy: int, prob_thresh: float, sin: np.ndarray[np.float64, 2],
                    cos: np.ndarray[np.float64, 2], col_prob_matrix: np.ndarray[np.float64, 2],
                    launched_sats: int, a_low: float, a_high: float, active_fraction: float,
                    plane: bool, num_workers: int) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2],
                                                            np.ndarray[np.float64, 2]]:


    # check if reusable indices are sufficient or if new rows have to be appended to sat_parameters
    indices = np.where(sat_parameters[:, -1] == -1)[0]
    if launched_sats - len(indices) > 0:
        num_new_parameters = int(launched_sats - len(indices))
    else:
        num_new_parameters = int(0)
    new_indices = np.empty(num_new_parameters, dtype=types.int64)
    for ii in range(num_new_parameters):
        new_indices[ii] = sat_parameters.shape[0] + ii

    # indices of all newly launched satellites
    extended_indices = np.empty(indices.shape[0] + new_indices.shape[0], dtype=types.int64)
    extended_indices[:indices.shape[0]] = indices
    extended_indices[indices.shape[0]:] = new_indices

    # parameters of newly launched satellites
    new_parameters, new_constants = initialize(launched_sats, a_low, a_high, active_fraction, plane)
    for ii, ind in enumerate(indices):
        sat_parameters[ind] = new_parameters[ii]
        sat_constants[ind] = new_constants[ii]

    sat_parameters = stack_arrays(sat_parameters, new_parameters[indices.shape[0]:])
    sat_constants = stack_arrays(sat_constants, new_constants[indices.shape[0]:])
    for ii, ind in enumerate(new_indices):
        sat_parameters[ind] = new_parameters[indices.shape[0] + ii]
        sat_constants[ind] = new_constants[indices.shape[0] + ii]

    indices = extended_indices

    results = []
    for worker in prange(num_workers):
        print()
        result = update(indices, sat_parameters, sat_constants, sigma, time_step, accuracy, sin, cos, prob_thresh,
                        worker, num_workers)
        results.append(result)

    for matrix in results:
        col_prob_matrix = stack_arrays(col_prob_matrix, matrix)

    return col_prob_matrix, sat_parameters, sat_constants


@jit(nopython=True)
def pool_update(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
                sigma: float, time_step: int, accuracy: int, prob_thresh: float, sin: np.ndarray[np.float64, 2],
                cos: np.ndarray[np.float64, 2], col_prob_matrix: np.ndarray[np.float64, 2],
                launched_sats: int, a_low: float, a_high: float, active_fraction: float,
                plane: bool, num_workers: int) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2],
                                                        np.ndarray[np.float64, 2]]:

    remove = [10, 443, 561, 777, 900]
    for ind in remove:
        sat_parameters[ind] = [0, 0, 0, 0, 0, 0, -1]
        sat_constants[ind] = [0, 0, 0, 0, 0, -1]

    # check if reusable indices are sufficient or if new rows have to be appended to sat_parameters
    indices = np.where(sat_parameters[:, -1] == -1)[0]
    if launched_sats - len(indices) > 0:
        num_new_parameters = int(launched_sats - len(indices))
    else:
        num_new_parameters = int(0)
    new_indices = np.empty(num_new_parameters, dtype=types.int64)
    for ii in range(num_new_parameters):
        new_indices[ii] = sat_parameters.shape[0] + ii

    # indices of all newly launched satellites
    extended_indices = np.empty(indices.shape[0] + new_indices.shape[0], dtype=types.int64)
    extended_indices[:indices.shape[0]] = indices
    extended_indices[indices.shape[0]:] = new_indices

    # parameters of newly launched satellites
    new_parameters, new_constants = initialize(launched_sats, a_low, a_high, active_fraction, plane)
    for ii, ind in enumerate(indices):
        sat_parameters[ind] = new_parameters[ii]
        sat_constants[ind] = new_constants[ii]

    sat_parameters = stack_arrays(sat_parameters, new_parameters[indices.shape[0]:])
    sat_constants = stack_arrays(sat_constants, new_constants[indices.shape[0]:])
    for ii, ind in enumerate(new_indices):
        sat_parameters[ind] = new_parameters[indices.shape[0] + ii]
        sat_constants[ind] = new_constants[indices.shape[0] + ii]

    indices = extended_indices

    results = []
    with Pool() as pool:
        processes = []
        for worker in range(num_workers):
            p = pool.apply_async(update, args=[indices, sat_parameters, sat_constants, sigma, time_step, accuracy,
                                               sin, cos, prob_thresh, worker, num_workers])
            processes.append(p)

        for process in processes:
            result = process.get()
            results.append(result)

    for matrix in results:
        col_prob_matrix = stack_arrays(col_prob_matrix, matrix)

    return col_prob_matrix, sat_parameters, sat_constants
