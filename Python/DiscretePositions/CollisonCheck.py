import time
import numpy as np
from numba import jit


EMIN, EMAX = 0, 0.9
IMIN, IMAX = 0, np.pi
WMIN, WMAX = 0, 2 * np.pi
OMIN, OMAX = 0, 2 * np.pi
MMIN, MMAX = 0, 1000


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
    satPositions = np.empty((nrOfSats, 3))

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


def orbital_position(satParameter, tmax, accuracy=1):
    '''
    Calculates the position (x, y, z) of a satellite for given orbital
    parameters (a, e, ...) and time.

    Args:
        aa (float): semimajor axis
        ee (float): eccentricity
        ii (float): inclination
        ww (float): argument of periapsis
        Om (float): length of the ascending node
        M0 (float): starting point of the mean anomaly
        TT (float): orbital period
        absoluteTime (int): time as curve parameter
        accuracy (int): Number of decimals to be rounded to. Defaults to 1.

    Returns:
        position (1darray): Orbital position component wise as elements of
                            an array

    '''
    nrOfSats = satParameter.shape[0]
    position = np.empty((nrOfSats, tmax, 3))
    absoluteTime = np.linspace(0, tmax, tmax)

    for satNr in range(nrOfSats):
        print(f'Calculating sat number: {satNr}')
        aa = satParameter[satNr][0]
        ee = satParameter[satNr][1]
        ii = satParameter[satNr][2]
        ww = satParameter[satNr][3]
        Om = satParameter[satNr][4]
        M0 = satParameter[satNr][5]
        TT = satParameter[satNr][6]
        for tt in range(len(absoluteTime)):
            MM = M0 + 2 * np.pi / TT * absoluteTime[tt]
            EE = MM + ee * np.sin(MM) + 1/2 * ee ** 2 * np.sin(2 * MM)

            XX = aa * (np.cos(EE) - ee)
            YY = aa * np.sqrt(1 - ee ** 2) * np.sin(EE)

            P11 = (np.cos(Om) * np.cos(ww) - np.sin(Om) * np.cos(ii) * np.sin(ww))
            P12 = (- np.cos(Om) * np.sin(ww) - np.sin(Om) * np.cos(ii) * np.cos(ww))
            P21 = (np.sin(Om) * np.cos(ww) + np.cos(Om) * np.cos(ii) * np.sin(ww))
            P22 = (- np.sin(Om) * np.sin(ww) + np.cos(Om) * np.cos(ii) * np.cos(ww))
            P31 = np.sin(ii) * np.sin(ww)
            P32 = np.sin(ii) * np.cos(ww)

            xx = XX * P11 + YY * P12
            yy = XX * P21 + YY * P22
            zz = XX * P31 + YY * P32

            position[satNr][tt][0] = xx
            position[satNr][tt][1] = yy
            position[satNr][tt][2] = zz

    position = np.around(position, decimals=accuracy)

    return position

tmax = 1000
satParams = initialize(10000, 10000, tmax)
pos = orbital_position(satParams, tmax)
print(pos[5])