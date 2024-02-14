import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def plot_orbits(parameters, indices):
    '''
    Plots a set orbit depending on thier respective orbital parameters.

    Args:
        parameters (2darray): Orbital elements of satellites on orbits..

    '''
    EE = np.linspace(0, 2 * np.pi, 1000)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for nn in indices[0:2]:

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

    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_zlim(zmin, zmax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.axis('off')

    save_path = os.path.join("C:\\Users\\jlhub\\Documents\\Studium\\Masterarbeit\\HubaldModell\\HubaldModel\\Python\\Plotting", "zoomed_orbits.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)
    #plt.close(fig)  # Close the figure


def plot_points(directory, sample_percentage=100):
    parameter_directory = os.path.join(directory, "satParameters.csv")
    probability_directory = os.path.join(directory, "probabilityMatrix.csv")
    satellite_parameters = np.genfromtxt(parameter_directory, delimiter=',')
    probability_matrix = np.genfromtxt(probability_directory, delimiter=',')

    number_endagered_sats = probability_matrix.shape[0]
    yellow_level = []
    red_level = []
    for jj in range(number_endagered_sats):
        if probability_matrix[jj][-1] > 1 * 10 ** (-5):
            red_level.append(int(probability_matrix[jj][0]))
            red_level.append(int(probability_matrix[jj][1]))
        else:
            yellow_level.append(int(probability_matrix[jj][0]))
            yellow_level.append(int(probability_matrix[jj][1]))

    number_of_satellites = satellite_parameters.shape[0]
    bound = int(number_of_satellites * (sample_percentage / 100))
    number_of_satellites = satellite_parameters.shape[0]
    satellite_positions = np.empty((number_of_satellites, 3))

    for jj in range(bound):
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
    ax.scatter(xpos, ypos, zpos, color='grey', alpha=0.3, label='low danger')

    sample_percentage = sample_percentage * 1
    bound = int(len(yellow_level) * (sample_percentage / 100))
    yellow_level = yellow_level[0:bound]
    yellow_positions = np.empty((len(yellow_level), 3))
    for jj, sat in enumerate(yellow_level):
        aa = satellite_parameters[sat][0]
        ee = satellite_parameters[sat][1]
        ii = satellite_parameters[sat][2]
        ww = satellite_parameters[sat][3]
        Om = satellite_parameters[sat][4]
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

        yellow_positions[jj][0] = xx
        yellow_positions[jj][1] = yy
        yellow_positions[jj][2] = zz

    xpos_y = yellow_positions[:, 0]
    ypos_y = yellow_positions[:, 1]
    zpos_y = yellow_positions[:, 2]

    ax.scatter(xpos_y, ypos_y, zpos_y, color='yellow', label='high danger')

    bound = int(len(red_level) * (sample_percentage / 100))
    red_level = red_level[0:bound]
    red_positions = np.empty((len(red_level), 3))
    for jj, sat in enumerate(red_level):
        aa = satellite_parameters[sat][0]
        ee = satellite_parameters[sat][1]
        ii = satellite_parameters[sat][2]
        ww = satellite_parameters[sat][3]
        Om = satellite_parameters[sat][4]
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

        red_positions[jj][0] = xx
        red_positions[jj][1] = yy
        red_positions[jj][2] = zz

    xpos_r = red_positions[:, 0]
    ypos_r = red_positions[:, 1]
    zpos_r = red_positions[:, 2]

    ax.scatter(xpos_r, ypos_r, zpos_r, color='red', label='critical danger')

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
    #ax.set_title('Satellite Positions')

    ax.axis('off')  # Turn off the coordinate system
    ax.legend()

    save_path = os.path.join("C:\\Users\\jlhub\\Documents\\Studium\\Masterarbeit\\HubaldModell\\HubaldModel\\Python\\Plotting", "orbital_positions.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close(fig)  # Close the figure

    return satellite_parameters, satellite_positions, red_level, yellow_level


if __name__ == '__main__':
    par_directory = os.path.abspath("C:/Users/jlhub/Documents/Studium/Masterarbeit/HubaldModell/HubaldModel/Python/Multiprocessing/output/Dec_04_10-10-16_2023/")
    sat_par, sat_pos, red, yellow = plot_points(par_directory, 2)
    plot_orbits(sat_par, red)
