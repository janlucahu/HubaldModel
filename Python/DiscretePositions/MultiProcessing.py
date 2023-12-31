'''
Attempt of simulating the kessler syndrome, searching for a distribution of
collision numbers depending on satellite density and critical values of
said density'''
# pylint: disable=invalid-name


import time
import numpy as np
import multiprocessing as mp
from numba import jit


EMIN, EMAX = 0, 0.9
IMIN, IMAX = 0, np.pi
WMIN, WMAX = 0, 2 * np.pi
OMIN, OMAX = 0, 2 * np.pi
MMIN, MMAX = 0, 1000

directory = 'C:\\Users\\jlhub\\Documents\\Studium\\Masterarbeit\\HubaldModell\\HubaldModel\\Python\\DiscretePositions\\Data\\'


@jit(nopython=True)
def orbital_position(satParameter, absoluteTime):
    '''
    Calculates the position (x, y, z) of a satellite for given orbital
    parameters (a, e, ...) and time.

    Args:
        satParameter ()
        absoluteTime (int): time as curve parameter
        accuracy (int): Number of decimals to be rounded to. Defaults to 1.

    Returns:
        satPosition (2darray): Orbital position component wise as elements of
                            an array

    '''
    satPosition = np.empty((satParameter.shape[0], 3))

    for satNr in range(satParameter.shape[0]):
        aa = satParameter[satNr][0]
        ee = satParameter[satNr][1]
        ii = satParameter[satNr][2]
        ww = satParameter[satNr][3]
        Om = satParameter[satNr][4]
        M0 = satParameter[satNr][5]
        TT = satParameter[satNr][6]

        MM = M0 + 2 * np.pi / TT * absoluteTime
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

        satPosition[satNr][0] = xx
        satPosition[satNr][1] = yy
        satPosition[satNr][2] = zz

    return satPosition


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
        TT = np.random.randint(3600, 10800)

        satParameter[satNr][0] = aa
        satParameter[satNr][1] = ee
        satParameter[satNr][2] = ii
        satParameter[satNr][3] = ww
        satParameter[satNr][4] = Om
        satParameter[satNr][5] = M0
        satParameter[satNr][6] = TT

    return satParameter


def check_collision(satPositions):
    '''
    Counts the instances of duplicate rows in an array

    Args:
        satPositions (2darray): Positions for each satellite. Columns depict
                                satellite number, rows positional components.

    Returns:
        int: Number of reappearing rows in input array

    '''
    unq, index, counter = np.unique(satPositions, axis=0, return_index=True,
                                    return_counts=True)
    collisionCheck = counter > 1
    if True in collisionCheck:
        uniqueCollisions = len(counter[collisionCheck])
        return uniqueCollisions
    else:
        return 0


def kessler_simulation(nrOfSats, size, tmax, accuracy=1, plane=True, run=None):
    '''
    Simulates the movement of a number of satellites in a system of given size
    moving around a focus point. If satellites collide, another satellite with
    random orbital parameters is added to the system. The number of collisions
    is counted and printed in the end.

    Args:
        nrOfSats (int): Number of starting satellites
        size (int): System size
        tmax (int): Number of iterations, influencing the orbital period
        accuracy (int): Number of decimals to be rounded to. Defaults to 1.
        plane (bool): Choose between a plane orbit or 3d orbit

    Returns:
        nrOfCollisions (int): Number of collisions encountered in the
                              simulation

    '''
    int1 = time.time()

    satPositions = np.empty((nrOfSats, 3))
    satParameter = initialize(nrOfSats, size, tmax, accuracy,
                                            plane)
    nrOfCollisions = 0

    print(f'Starting satellites: {nrOfSats}, Size: {size}, Run Nr.: {run}')
    for sec in range(tmax):

        satPositions = np.around(satPositions, decimals=accuracy)
        uniqueCollisions = check_collision(satPositions)
        for collisions in range(uniqueCollisions):
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
            newParameter = np.array((aa, ee, ii, ww, Om, M0, TT))

            satParameter = np.vstack((satParameter, newParameter))

            satPositions = orbital_position(satParameter, sec)
            nrOfCollisions += 1

        progress = np.around(sec / tmax * 100, decimals=2)
        print(f'\r{progress} %', end='', flush=True)

    int2 = time.time()
    print('')
    print('Number of Collisions: ', nrOfCollisions, ' after ',
          np.around(int2 - int1, decimals=2), ' seconds')
    print('***************************************************')
    return nrOfCollisions


def runSatellites(startingSatellites, stepSize, stepNumber, systemSize,
                  averageSize, tmax, accuracy=1, plane=True, file='data'):
    '''
    Runs multiple simulations of satellite motion and collisions for different
    satellite densities. Prints lists of data for the used starting satellites,
    density and encountered collisions, averaged over a number of calculations.

    Args:
        startingSatellites (int): Number of satellites for first simulation
        stepSize (int): Size of steps for increasing number of satellites
        stepNumber (int): Number of steps for increasing number of satellites
        systemSize (int): System size
        averageSize (int): Number of runs for each number of satellites,
                           collisions are averaged over these
        tmax (int): Number of iterations, influencing the orbital period
        accuracy (int): Number of decimals to be rounded to. Defaults to 1.
        plane (bool): Choose between a plane orbit or 3d orbit
        file (string): Name of the file the data is writen into

    Returns:
        None.

    '''
    drt = directory
    filename = drt + file + '.txt'
    nrOfSatellites = []
    satelliteDensity = []
    averageNrOfCollisions = []
    calculationTime = []
    for step in range(stepNumber):
        int1 = time.time()

        satellites = startingSatellites + stepSize * step
        print('Starting satellites: ', satellites)
        collisions = []
        for ii in range(averageSize):
            cols = kessler_simulation(satellites, systemSize, tmax, accuracy,
                                      plane, ii+1)
            collisions.append(cols)

        nrOfSatellites.append(satellites)
        if plane:
            satelliteDensity.append(satellites / (systemSize ** 2))
        else:
            satelliteDensity.append(satellites / (systemSize ** 3))
        averageNrOfCollisions.append(np.sum(collisions) / len(collisions))

        int2 = time.time()
        interval = np.around(int2 - int1, decimals=3)
        calculationTime.append(interval)

        with open(filename, "w") as output:
            output.write('Satellites:\n')
            output.write(str(nrOfSatellites))
            output.write('\nDensity:\n')
            output.write(str(satelliteDensity))
            output.write('\nCollisions:\n')
            output.write(str(averageNrOfCollisions))
            output.write('\nCalculation time:\n')
            output.write(str(calculationTime))

    print('Satellites: ', nrOfSatellites)
    print('Satellite density: ', satelliteDensity)
    print('Nr. of collisions: ', averageNrOfCollisions)


def main():
    '''
    Main function

    Returns:
        None.

    '''
    startingTime = time.time()
    kessler_simulation(1000, 1000000, 86400)
    endingTime = time.time()
    print(f'Process finished in {endingTime - startingTime} seconds')


if __name__ == '__main__':
    main()
