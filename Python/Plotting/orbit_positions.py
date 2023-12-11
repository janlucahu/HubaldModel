import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points(parameter_directory, sample_percentage=100):
    satellite_parameters = np.genfromtxt(parameter_directory, delimiter=',')
    number_of_satellites = satellite_parameters.shape[0]
    bound = int(number_of_satellites * (sample_percentage / 100))
    satellite_parameters = satellite_parameters[0:bound, :]
    number_of_satellites = satellite_parameters.shape[0]
    satellite_positions = np.empty((number_of_satellites, 3))

    for jj in range(number_of_satellites):
        aa = satellite_parameters[jj][0]
        ee = satellite_parameters[jj][1]
        ii = satellite_parameters[jj][2]
        ww = satellite_parameters[jj][3]
        Om = satellite_parameters[jj][4]
        M0 = np.random.randint(0, 1000)

        MM = M0
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

        satellite_positions[jj][0] = xx
        satellite_positions[jj][1] = yy
        satellite_positions[jj][2] = zz

    xpos = satellite_positions[:, 0]
    ypos = satellite_positions[:, 1]
    zpos = satellite_positions[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xpos, ypos, zpos, color='grey')

    radius = 6357000
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Satellite Positions')

    plt.show()

    return satellite_parameters, satellite_positions


if __name__ == '__main__':
    par_directory = os.path.abspath("C:/Users/jlhub/Documents/Studium/Masterarbeit/HubaldModell/HubaldModel/Python/Multiprocessing/output/Dec_04_10-10-16_2023/satParameters.csv")
    sat_par, sat_pos = plot_points(par_directory, 5)
