import numpy as np
from numba import jit


@jit(nopython=True)
def constants(sat_parameters: np.ndarray[np.float64, 2]) -> np.ndarray[np.float64, 1]:
    i1 = sat_parameters[2]
    w1 = sat_parameters[3]
    O1 = sat_parameters[4]

    P11_1 = np.cos(O1) * np.cos(w1) - np.sin(O1) * np.cos(i1) * np.sin(w1)
    P12_1 = - np.cos(O1) * np.sin(w1) - np.sin(O1) * np.cos(i1) * np.cos(w1)
    P21_1 = np.sin(O1) * np.cos(w1) + np.cos(O1) * np.cos(i1) * np.sin(w1)
    P22_1 = - np.sin(O1) * np.sin(w1) + np.cos(O1) * np.cos(i1) * np.cos(w1)
    P31_1 = np.sin(i1) * np.sin(w1)
    P32_1 = np.sin(i1) * np.cos(w1)

    entries = [P11_1, P12_1, P21_1, P22_1, P31_1, P32_1]
    sat_constants = np.array(entries)

    return sat_constants


@jit(nopython=True)
def initialize(num_sats: int, a_low: float, a_high: float, active_fraction: float,
               plane: bool) -> tuple[np.ndarray[np.float64, 2], np.ndarray[np.float64, 2]]:

    EMIN, EMAX = float(0), float(0.3)
    IMIN, IMAX = float(0), float(0.5 * np.pi)
    WMIN, WMAX = float(0), float(1.0 * np.pi)
    OMIN, OMAX = float(0), float(1.0 * np.pi)

    sat_parameters = np.empty((num_sats, 7))
    sat_constants = np.empty((num_sats, 6))

    AMIN = a_low
    AMAX = a_high
    signs = np.array([-1, 1])
    for sat in range(num_sats):
        ee = np.random.uniform(EMIN, EMAX)
        aa = np.random.uniform(AMIN, AMAX)

        if plane:
            ii = 0
        else:
            ii = np.random.uniform(IMIN, IMAX)

        ww = np.random.uniform(WMIN, WMAX)
        Om = np.random.uniform(OMIN, OMAX)
        sign = signs[np.random.randint(0, 2)]
        CEarth = 9.91 * 10 ** (-14)
        TT = sign * np.sqrt(CEarth * aa ** 3)
        if np.random.rand() < active_fraction:
            active = 1
        else:
            active = 0

        sat_parameters[sat][0] = aa
        sat_parameters[sat][1] = ee
        sat_parameters[sat][2] = ii
        sat_parameters[sat][3] = ww
        sat_parameters[sat][4] = Om
        sat_parameters[sat][5] = TT
        sat_parameters[sat][6] = active

        sat_constants[sat] = constants(sat_parameters[sat])

    return sat_parameters, sat_constants


@jit(nopython=True)
def calculate_trig(accuracy: int, mode: str) -> np.ndarray[np.float64, 2]:
    EE = np.empty((accuracy, accuracy))
    for ii in range(accuracy):
        for jj in range(accuracy):
            EE[ii][jj] = jj * 2 * np.pi / accuracy

    trig_E = np.empty((accuracy, accuracy))
    if mode == "s":
        for ii in range(accuracy):
            for jj in range(accuracy):
                trig_E[ii][jj] = np.sin(EE[ii][jj])
    elif mode == "c":
        for ii in range(accuracy):
            for jj in range(accuracy):
                trig_E[ii][jj] = np.cos(EE[ii][jj])
    else:
        raise ValueError("No valid trig function.")

    return trig_E


@jit(nopython=True)
def find_minimum(parameters1: np.ndarray[np.float64, 1], parameters2: np.ndarray[np.float64, 1],
                 constants1: np.ndarray[np.float64, 1], constants2: np.ndarray[np.float64, 1], accuracy: int,
                 sin: np.ndarray[np.float64, 2], cos: np.ndarray[np.float64, 2]) -> float:

    a1, a2 = parameters1[0], parameters2[0]
    e1, e2 = parameters1[1], parameters2[1]

    P11_1, P11_2 = constants1[0], constants2[0]
    P12_1, P12_2 = constants1[1], constants2[1]
    P21_1, P21_2 = constants1[2], constants2[2]
    P22_1, P22_2 = constants1[3], constants2[3]
    P31_1, P31_2 = constants1[4], constants2[4]
    P32_1, P32_2 = constants1[5], constants2[5]

    X1 = a1 * (cos - e1)
    Y1 = a1 * np.sqrt(1 - e1 ** 2)

    x1 = X1 * P11_1 + Y1 * P12_1
    y1 = X1 * P21_1 + Y1 * P22_1
    z1 = X1 * P31_1 + Y1 * P32_1

    X2 = a2 * (cos - e2)
    Y2 = a2 * np.sqrt(1 - e2 ** 2) * sin

    x2 = X2 * P11_2 + Y2 * P12_2
    y2 = X2 * P21_2 + Y2 * P22_2
    z2 = X2 * P31_2 + Y2 * P32_2

    dist_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2

    minimum_distance = np.min(dist_squared)

    return minimum_distance


@jit(nopython=True)
def collision_probability(sat_parameters: np.ndarray[np.float64, 2], sat_constants: np.ndarray[np.float64, 2],
                          sat1: int, sat2: int, sigma: float, time_step: int, accuracy: int,
                          sin: np.ndarray[np.float64, 2], cos: np.ndarray[np.float64, 2]) -> float:
    col_prob = float(0.0)
    month_to_seconds = float(30 * 24 * 60 * 60)

    if sat_parameters[sat1][6] != -1 and sat_parameters[sat2][6] != -1:
        active_satellite = bool(sat_parameters[sat1][6] + sat_parameters[sat2][6])
        synodic_period = float(1 / np.abs(1 / sat_parameters[sat1][5] + 1 / sat_parameters[sat2][5]))
        num_approaches = float(np.floor(time_step * month_to_seconds / synodic_period))

        parameters1 = sat_parameters[sat1]
        parameters2 = sat_parameters[sat2]
        constants1 = sat_constants[sat1]
        constants2 = sat_constants[sat2]

        min_distance = find_minimum(parameters1, parameters2, constants1, constants2, accuracy, sin, cos)

        if active_satellite:
            factor = float(5 * 10 ** (-5))
        else:
            factor = float(5 * 10 ** (-1))

        col_prob_per_approach = float(factor * np.exp(-min_distance ** 2 / (2 * sigma ** 2)))
        col_prob = 1 - (1 - col_prob_per_approach) ** num_approaches

    return col_prob
