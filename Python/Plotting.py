'''Routines dedicated to plotting of simple data'''

import matplotlib.pyplot as plt
import numpy as np
import ast
"""
directory = 'C:\\Users\\jlhub\\Documents\\Studium\\HubaldModel\\HubaldKessler\\data\\size10000\\'

with open(directory + 'sats10000_size10000.txt') as file:
    for i, line in enumerate(file):
        if i == 1:
            sats = line
        if i == 3:
            dens = line
        if i == 5:
            cols = line
        if i == 7:
            time = line"""

def get_data(file):
    with open(directory + file) as fp:
        for i, line in enumerate(fp):
            if i == 1:
                sats = ast.literal_eval(line)
            if i == 3:
                dens = ast.literal_eval(line)
            if i == 5:
                cols = ast.literal_eval(line)
            if i == 7:
                time = ast.literal_eval(line)
    return sats, dens, cols, time


def merge_data(nrOfFiles, startSats, size):
    satellites = []
    density = []
    collisions = []
    calcTime = []

    for files in range(nrOfFiles):
        startingSats = startSats ** (files + 1)
        size = 10000

        file = 'sats' + str(startingSats) + '_size' + str(size) + '.txt'

        sats, dens, cols, time = get_data(file)

        satellites += sats
        density += dens
        collisions += cols
        calcTime += time

    return satellites, density, collisions, calcTime


def plot_lists(xx, yy, scatter=False, log=False, logValues=False, fit=False,
               xlab='x', ylab='y'):
    if logValues:
        for ii in range(len(yy)):
            if yy[ii] == 0:
                yy[ii] = 0.01
        xx = [np.log10(value) for value in xx]
        yy = [np.log10(value) for value in yy]
    if scatter:
        plt.scatter(xx,yy)
    else:
        plt.plot(xx, yy)
    if log:
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.01, 50000)
        plt.xlim(0.000001, 0.002)
    if fit:
        coef = np.polyfit(xx,yy,1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(xx, yy, 'yo', xx, poly1d_fn(xx), '--k')
        plt.legend(coef)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def main():
    """
    sats, dens, cols, calcTime = merge_data(4, 10, 10000)

    plot_lists(dens, cols, scatter=True, log=True, logValues=False, fit=False,
               xlab='Satellite density', ylab='Number of collisions')
    plot_lists(dens[20::], cols[20::], scatter=True, log=False, logValues=True,
               fit=True,
               xlab='Log Satellite density', ylab='Log Number of collisions')

    plot_lists(dens, calcTime, fit=True,
                xlab='Satellite density', ylab='Calculation Time in s')
                """
    sats = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000]
    single = [12.486984968185425, 39.39278316497803, 242.03981280326843, 962.6959290504456, 2166.925658226013, 3831.294944047928, 8622.896792888641, 15316.554593086243, 23922.916858911514, 54102.562933921814, 96028.11221194267]
    multi = [9.93146800994873, 15.893611907958984, 59.121577978134155, 210.30724382400513, 467.563912153244, 821.8911366462708, 1833.5937569141388, 3240.694380044937, 5045.069607019424, 11282.964400053024, 20048.669051885605]

    plt.plot(sats, single, label='single')
    plt.plot(sats, multi, label='multi')
    plt.xlabel("Number of satellites")
    plt.ylabel("Calculation time (s)")
    plt.title("Calculation time comparison of single and multi-core computation")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
