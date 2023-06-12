import numpy as np
import time
import os
from numba import jit


EMIN, EMAX = 0, 0.9
IMIN, IMAX = 0, np.pi
WMIN, WMAX = 0, 2 * np.pi
OMIN, OMAX = 0, 2 * np.pi
MMIN, MMAX = 0, 1000


@jit(nopython=True)
def initialize(nrOfSats, size, tmax, accuracy=1, plane=True):
    '''
    Inititalizes a system of satellites orbiting around a focus point. The size
    of the system specifies the bounderies, which the satellites won't pass

    Args:
        nrOfSats (int): Number of satellites to be initialized
        size (int): Size of the system
        tmax (int): Maximum time, influencing the orbital period
        accuracy (int): Number of decimals to be rounded to. Defaults to 1.
        plane (bool): Choose between a plane orbit or 3d orbit

    Returns:
        satParameter (2darray): Orbital parameters for each Satellite. Columns
                                depict satellite number, rows orbital
                                parameters.
        satPositions (2darray): Positions for each satellite. Columns depict
                                satellite number, rows positional components.

    '''
    satParameter = np.empty((nrOfSats, 7))

    for satNr in range(nrOfSats):
        ee = np.random.uniform(EMIN, EMAX)
        # upper limit assures that no satellite is out of bounds
        aa = np.random.uniform(0.1* size, (size / 2) / (1 + ee))

        if plane:
            ii = 0
        else:
            ii = np.random.uniform(IMIN, IMAX)

        ww = np.random.uniform(WMIN, WMAX)
        Om = np.random.uniform(OMIN, OMAX)
        M0 = np.random.randint(MMIN, MMAX)
        TT = np.random.randint(1/5 * tmax, tmax)

        satParameter[satNr][0] = aa
        satParameter[satNr][1] = ee
        satParameter[satNr][2] = ii
        satParameter[satNr][3] = ww
        satParameter[satNr][4] = Om
        satParameter[satNr][5] = M0
        satParameter[satNr][6] = TT

    return satParameter


@jit(nopython=True)
def closest_distance(a1, e1, i1, Omega1, omega1, a2, e2, i2, Omega2, omega2):
    """
    Calculates the closest distance between two elliptical orbits given their orbital elements.

    Parameters:
    -----------
    a1 : float or numpy.ndarray
        Semi-major axis of the first orbit (in AU).
    e1 : float or numpy.ndarray
        Eccentricity of the first orbit.
    i1 : float or numpy.ndarray
        Inclination of the first orbit (in degrees).
    Omega1 : float or numpy.ndarray
        Longitude of ascending node of the first orbit (in degrees).
    omega1 : float or numpy.ndarray
        Argument of periapsis of the first orbit (in degrees).
    a2 : float or numpy.ndarray
        Semi-major axis of the second orbit (in AU).
    e2 : float or numpy.ndarray
        Eccentricity of the second orbit.
    i2 : float or numpy.ndarray
        Inclination of the second orbit (in degrees).
    Omega2 : float or numpy.ndarray
        Longitude of ascending node of the second orbit (in degrees).
    omega2 : float or numpy.ndarray
        Argument of periapsis of the second orbit (in degrees).

    Returns:
    --------
    float or numpy.ndarray
        Closest distance between the two orbits (in AU).
    """

    # Convert angles to radians
    i1 = np.radians(i1)
    Omega1 = np.radians(Omega1)
    omega1 = np.radians(omega1)
    i2 = np.radians(i2)
    Omega2 = np.radians(Omega2)
    omega2 = np.radians(omega2)

    # Calculate the relative inclination and longitude of ascending node
    dOmega = Omega2 - Omega1
    di = i2 - i1

    # Calculate the semi-latus rectum and periapsis distance of each orbit
    p1 = a1 * (1 - e1**2)
    p2 = a2 * (1 - e2**2)
    r1 = p1 / (1 + e1 * np.cos(omega1))
    r2 = p2 / (1 + e2 * np.cos(omega2))

    # Calculate the distance between the two foci of each orbit
    f1 = a1 * e1
    f2 = a2 * e2
    df = f2 - f1

    # Calculate the cosine and sine of the angles used in the distance formula
    cos_theta = np.cos(dOmega) * np.cos(omega2) * np.cos(omega1) + np.sin(dOmega) * np.sin(omega2) * np.sin(omega1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    cos_phi = np.cos(di) * (r1 + r2 * cos_theta) / df - r2 * sin_theta * np.sin(di) / df
    sin_phi = np.sqrt(1 - np.where(cos_phi**2 < 1, 1 - cos_phi**2, 0))

    # Calculate the closest distance between the two orbits
    distance = df * cos_phi - np.sqrt(np.where(r1**2 - (r2 * sin_theta * sin_phi)**2 > 0, r1**2 - (r2 * sin_theta * sin_phi)**2, 0))

    return distance


def distance_matrix(nrOfSats, satParameters):
    dtype = np.float64
    shape = (nrOfSats, nrOfSats)

    filename = 'distances.dat'
    filepath = os.path.abspath(filename)

    fp = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)

    for sat1 in range(nrOfSats):
        progress = np.around(sat1 / nrOfSats * 100, decimals=2)
        print(f'\r{progress} %', end='', flush=True)
        a1 = satParameters[sat1][0]
        e1 = satParameters[sat1][1]
        i1 = satParameters[sat1][2]
        w1 = satParameters[sat1][3]
        O1 = satParameters[sat1][4]
        for sat2 in range(sat1):
            a2 = satParameters[sat2][0]
            e2 = satParameters[sat2][1]
            i2 = satParameters[sat2][2]
            w2 = satParameters[sat2][3]
            O2 = satParameters[sat2][4]

            fp[sat1, sat2] = closest_distance(a1, e1, i1, O1, w1, a2,
                                                     e2, i2, O2, w2)

    del fp  # Close the memory-mapped file

    return filepath


sats = 1000
start = time.time()
satParameters = initialize(sats, 1000000, 86400)
d_matrix = distance_matrix(sats, satParameters)
finish = time.time()

print('\nProcess finished after: ', finish - start)
