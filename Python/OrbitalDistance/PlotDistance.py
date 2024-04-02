import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


EMIN, EMAX = 0, 0.7
IMIN, IMAX = 0, np.pi
WMIN, WMAX = 0, 2 * np.pi
OMIN, OMAX = 0, 2 * np.pi


def initialize(nrOfSats, size, plane=False):
    '''
    Inititalizes a system of satellites orbiting around a focus point. The size
    of the system specifies the bounderies, which the satellites won't pass.

    Args:
        nrOfSats (int): Number of satellites to be initialized.
        size (int): Size of the system.
        plane (bool, optional): Choose between a plane orbit or 3d orbit.
                                Defaults to false (3d orbit).

    Returns:
        satParameter (2darray): Orbital parameters for each Satellite. Columns
                                depict satellite number, rows orbital
                                parameters.

    '''
    satParameter = np.empty((nrOfSats, 5))

    for satNr in range(nrOfSats):
        ee = np.random.uniform(EMIN, EMAX)
        # upper limit assures that no satellite is out of bounds
        aa = np.random.uniform(0.2* size, (size / 2) / (1 + ee))

        if plane:
            ii = 0
        else:
            ii = np.random.uniform(IMIN, IMAX)

        ww = np.random.uniform(WMIN, WMAX)
        Om = np.random.uniform(OMIN, OMAX)

        satParameter[satNr][0] = aa
        satParameter[satNr][1] = ee
        satParameter[satNr][2] = ii
        satParameter[satNr][3] = ww
        satParameter[satNr][4] = Om

    return satParameter


def constants2(parameters1, parameters2):
    i1, i2 = parameters1[2], parameters2[2]
    w1, w2 = parameters1[3], parameters2[3]
    O1, O2 = parameters1[4], parameters2[4]

    P11_1 = np.cos(O1) * np.cos(w1) - np.sin(O1) * np.cos(i1) * np.sin(w1)
    P12_1 = - np.cos(O1) * np.sin(w1) - np.sin(O1) * np.cos(i1) * np.cos(w1)
    P21_1 = np.sin(O1) * np.cos(w1) + np.cos(O1) * np.cos(i1) * np.sin(w1)
    P22_1 = - np.sin(O1) * np.sin(w1) + np.cos(O1) * np.cos(i1) * np.cos(w1)
    P31_1 = np.sin(i1) * np.sin(w1)
    P32_1 = np.sin(i1) * np.cos(w1)

    P11_2 = np.cos(O2) * np.cos(w2) - np.sin(O2) * np.cos(i2) * np.sin(w2)
    P12_2 = - np.cos(O2) * np.sin(w2) - np.sin(O2) * np.cos(i2) * np.cos(w2)
    P21_2 = np.sin(O2) * np.cos(w2) + np.cos(O2) * np.cos(i2) * np.sin(w2)
    P22_2 = - np.sin(O2) * np.sin(w2) + np.cos(O2) * np.cos(i2) * np.cos(w2)
    P31_2 = np.sin(i2) * np.sin(w2)
    P32_2 = np.sin(i2) * np.cos(w2)

    const1 = (P11_1, P12_1, P21_1, P22_1, P31_1, P32_1)
    const2 = (P11_2, P12_2, P21_2, P22_2, P31_2, P32_2)

    return const1, const2


def find_minimum(parameters1, parameters2, acc=100, repetitions=3):
    E_1 = np.linspace(0, 2 * np.pi, acc)
    E_2 = np.linspace(0, 2 * np.pi, acc)
    E1, E2 = np.meshgrid(E_1, E_2)

    a1, a2 = parameters1[0], parameters2[0]
    e1, e2 = parameters1[1], parameters2[1]

    const1, const2 = constants2(parameters1, parameters2)
    P11_1, P12_1, P21_1, P22_1, P31_1, P32_1 = const1
    P11_2, P12_2, P21_2, P22_2, P31_2, P32_2 = const2

    X1 = a1 * (np.cos(E1) - e1)
    Y1 = a1 * np.sqrt(1 - e1 ** 2) * np.sin(E1)

    x1 = X1 * P11_1 + Y1 * P12_1
    y1 = X1 * P21_1 + Y1 * P22_1
    z1 = X1 * P31_1 + Y1 * P32_1

    X2 = a2 * (np.cos(E2) - e2)
    Y2 = a2 * np.sqrt(1 - e2 ** 2) * np.sin(E2)

    x2 = X2 * P11_2 + Y2 * P12_2
    y2 = X2 * P21_2 + Y2 * P22_2
    z2 = X2 * P31_2 + Y2 * P32_2

    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    minRow = None
    minCol = None

    minDistances = []
    minIndices = []
    minCoordinates = []

    for rep in range(repetitions):
        if minRow is None and minCol is None:
            minDistance = round(np.min(dist), 2)
            minIndex = np.argmin(dist)
            minRow, minCol = np.unravel_index(minIndex, dist.shape)

            minDistances.append(minDistance)
            minIndices.append([minRow, minCol])
            minCoordinates.append([E1[0][minCol], E2[minRow][0]])
        else:
            ival = 2 / (10 ** rep)
            E_1 = np.linspace(E_1[minCol] - ival, E_1[minCol] + ival, acc)
            E_2 = np.linspace(E_2[minRow] - ival, E_2[minRow] + ival, acc)
            E1, E2 = np.meshgrid(E_1, E_2)
            X1 = a1 * (np.cos(E1) - e1)
            Y1 = a1 * np.sqrt(1 - e1 ** 2) * np.sin(E1)

            x1 = X1 * P11_1 + Y1 * P12_1
            y1 = X1 * P21_1 + Y1 * P22_1
            z1 = X1 * P31_1 + Y1 * P32_1

            X2 = a2 * (np.cos(E2) - e2)
            Y2 = a2 * np.sqrt(1 - e2 ** 2) * np.sin(E2)

            x2 = X2 * P11_2 + Y2 * P12_2
            y2 = X2 * P21_2 + Y2 * P22_2
            z2 = X2 * P31_2 + Y2 * P32_2

            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

            minDistance = round(np.min(dist), 2)
            minIndex = np.argmin(dist)
            minRow, minCol = np.unravel_index(minIndex, dist.shape)

            minDistances.append(minDistance)
            minIndices.append([minRow, minCol])
            minCoordinates.append([E1[minCol][0], E2[minRow][0]])

    return minDistances, minIndices, minCoordinates


def plot_distance(parameters1, parameters2, acc=100):
    E_1 = np.linspace(0, 2 * np.pi, acc)
    E_2 = np.linspace(0, 2 * np.pi, acc)
    E1, E2 = np.meshgrid(E_1, E_2)

    a1, a2 = parameters1[0], parameters2[0]
    e1, e2 = parameters1[1], parameters2[1]

    const1, const2 = constants2(parameters1, parameters2)
    P11_1, P12_1, P21_1, P22_1, P31_1, P32_1 = const1
    P11_2, P12_2, P21_2, P22_2, P31_2, P32_2 = const2

    parStrings = [['a1', 'a2'], ['e1', 'e2'], ['i1', 'i2'], ['w1', 'w2'],
                  ['O1', 'O2']]

    print('Orbital parameters:')
    for ii, par in enumerate(parStrings):
        par1 = round(parameters1[ii], 2)
        par2 = round(parameters2[ii], 2)
        print(par[0], f': {par1}', '    ', par[1], f': {par2}')
    print('')

    X1 = a1 * (np.cos(E1) - e1)
    Y1 = a1 * np.sqrt(1 - e1 ** 2) * np.sin(E1)

    x1 = X1 * P11_1 + Y1 * P12_1
    y1 = X1 * P21_1 + Y1 * P22_1
    z1 = X1 * P31_1 + Y1 * P32_1

    X2 = a2 * (np.cos(E2) - e2)
    Y2 = a2 * np.sqrt(1 - e2 ** 2) * np.sin(E2)

    x2 = X2 * P11_2 + Y2 * P12_2
    y2 = X2 * P21_2 + Y2 * P22_2
    z2 = X2 * P31_2 + Y2 * P32_2

    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)/1000

    minDistances, minIndices, minCoord = find_minimum(parameters1, parameters2, acc=acc, repetitions=3)
    minDistance = minDistances[-1]
    print('Minimum distances: ', minDistances)

    minPoints = []
    for ii, angles in enumerate(minCoord):
        minPoints.append([angles[0], angles[1], minDistances[ii]/1000])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(E1, E2, dist, cmap='coolwarm', vmin=dist.min(),
                           vmax=dist.max())

    # Add legend
    # ax.text(0.8, 0.8, minDistance,
    #         "Minimum distance in km: {:.2f}".format(minDistance),
    #         transform=ax.transAxes)
    ax.set_xlabel('E1', fontsize=12)
    ax.set_ylabel('E2', fontsize=12)
    ax.set_zlabel('Distance [km]', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    # Add points
    minPoints = np.array(minPoints)
    sc = ax.scatter(minPoints[:, 0], minPoints[:, 1], minPoints[:, 2],
                    c=minPoints[:, 2], cmap='copper', vmin=dist.min(),
                    vmax=dist.max(), marker='o')

    path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(path, 'MinimumDistance.png')
    plt.savefig(filepath, dpi=1200)
    plt.show()


def plot_orbits(parameters, indices):
    '''
    Plots a set orbit depending on thier respective orbital parameters.

    Args:
        parameters (2darray): Orbital elements of satellites on orbits..

    '''
    EE = np.linspace(0, 2 * np.pi, 1000)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for nn in indices:

        aa = parameters[nn][0]
        ee = parameters[nn][1]

        XX = aa * (np.cos(EE) - ee)
        YY = aa * np.sqrt(1 - ee ** 2) * np.sin(EE)

        const1, *_ = constants2(parameters[nn], parameters[nn])
        P11, P12, P21, P22, P31, P32 = const1

        xx = XX * P11 + YY * P12
        yy = XX * P21 + YY * P22
        zz = XX * P31 + YY * P32

        ax.plot(xx, yy, zz)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def distance_histogram(distanceMatrix, bins=50):
    counts, bins = np.histogram(distanceMatrix, bins=bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel('Closest approach distance in m')
    plt.ylabel('Number of occurance')
    plt.show()


def main():
    '''
    Main function.

    Returns:
        None.

    '''
    chooseParas = False
    plotOrbits = True

    if not chooseParas:
        parameters = initialize(10, 1_000_000)
    else:
        parameters = np.empty((2, 5))
        parameters[0] = [249074.88, 0.54, 2.08, 5.41, 5.07]
        parameters[1] = [279262.79, 0.57, 1.87, 3.06, 1.92]

    plot_distance(parameters[0], parameters[1], acc=20)
    if plotOrbits:
        indices = [0, 2, 4, 6, 8]
        plot_orbits(parameters, indices)


if __name__ == '__main__':
    main()
