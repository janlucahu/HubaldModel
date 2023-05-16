'''Routines dedicated to plotting of simple data'''

import matplotlib.pyplot as plt
import numpy as np
import ast

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
            time = line

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
    sats, dens, cols, calcTime = merge_data(4, 10, 10000)

    plot_lists(dens, cols, scatter=True, log=True, logValues=False, fit=False,
               xlab='Satellite density', ylab='Number of collisions')
    plot_lists(dens[20::], cols[20::], scatter=True, log=False, logValues=True,
               fit=True,
               xlab='Log Satellite density', ylab='Log Number of collisions')

    plot_lists(dens, calcTime, fit=True,
                xlab='Satellite density', ylab='Calculation Time in s')


if __name__ == '__main__':
    main()
