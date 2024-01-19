import numpy as np
from numba import jit


EMIN, EMAX = 0, 0.3
IMIN, IMAX = 0, 0.5 * np.pi
WMIN, WMAX = 0, 1.0 * np.pi
OMIN, OMAX = 0, 1.0 * np.pi
TMIN, TMAX = 1.5, 8


@jit(nopython=True)
def constants(satParameters):
    '''
    Calculate constants used to calculate the orbits of the satellites.

    Args:
        satParameters (1darray): Orbital parameters of the satellites.

    Returns:
        satConstants (1darray): Rotation matrix elements for position calculation in satellite frame.
    '''
    i1 = satParameters[2]
    w1 = satParameters[3]
    O1 = satParameters[4]

    P11_1 = np.cos(O1) * np.cos(w1) - np.sin(O1) * np.cos(i1) * np.sin(w1)
    P12_1 = - np.cos(O1) * np.sin(w1) - np.sin(O1) * np.cos(i1) * np.cos(w1)
    P21_1 = np.sin(O1) * np.cos(w1) + np.cos(O1) * np.cos(i1) * np.sin(w1)
    P22_1 = - np.sin(O1) * np.sin(w1) + np.cos(O1) * np.cos(i1) * np.cos(w1)
    P31_1 = np.sin(i1) * np.sin(w1)
    P32_1 = np.sin(i1) * np.cos(w1)

    entries = [P11_1, P12_1, P21_1, P22_1, P31_1, P32_1]
    satConstants = np.array(entries)

    return satConstants


@jit(nopython=True)
def initialize(nrOfSats, alimits, activeFraction, plane=False):
    '''
    Initializes random orbital parameters for a given number of satellites.

    Args:
        nrOfSats (int): Number of satellites for which the orbital parameters should be initialized.
        alimits (float, float): Upper and lower limit for the semi-major axes.
        activeFraction (float): Percentage of active satellites initialized.
        plane (bool): Only planar orbits (inclination = 0) are initialized if set to True.

    Returns:
        satParameter (2darray): Orbital parameters, period time and active status for each satellite.
        satConstants (2darray): Rotation matrix elements for position calculation for each satellite.
    '''
    satParameters = np.empty((nrOfSats, 7))
    satConstants = np.empty((nrOfSats, 6))

    AMIN, AMAX = alimits
    signs = np.array([-1, 1])
    for satNr in range(nrOfSats):
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
        if np.random.rand() < activeFraction:
            active = 1
        else:
            active = 0

        satParameters[satNr][0] = aa
        satParameters[satNr][1] = ee
        satParameters[satNr][2] = ii
        satParameters[satNr][3] = ww
        satParameters[satNr][4] = Om
        satParameters[satNr][5] = TT
        satParameters[satNr][6] = active

        satConstants[satNr] = constants(satParameters[satNr])

    return satParameters, satConstants
