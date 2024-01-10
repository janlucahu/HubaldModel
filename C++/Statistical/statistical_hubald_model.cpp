#include "statistical_hubald_model.h"

#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

// Constants
const double EMIN = 0.0;
const double EMAX = 0.3;
const double IMIN = 0.0;
const double IMAX = 0.5 * M_PI;
const double WMIN = 0.0;
const double WMAX = 1.0 * M_PI;
const double OMIN = 0.0;
const double OMAX = 1.0 * M_PI;

// Function to calculate constants
std::vector<double> constants(const std::vector<double>& sat_parameters) {
    double i1 = sat_parameters[2];
    double w1 = sat_parameters[3];
    double O1 = sat_parameters[4];

    double P11_1 = cos(O1) * cos(w1) - sin(O1) * cos(i1) * sin(w1);
    double P12_1 = -cos(O1) * sin(w1) - sin(O1) * cos(i1) * cos(w1);
    double P21_1 = sin(O1) * cos(w1) + cos(O1) * cos(i1) * sin(w1);
    double P22_1 = -sin(O1) * sin(w1) + cos(O1) * cos(i1) * cos(w1);
    double P31_1 = sin(i1) * sin(w1);
    double P32_1 = sin(i1) * cos(w1);

    std::vector<double> sat_constants = { P11_1, P12_1, P21_1, P22_1, P31_1, P32_1 };
    return sat_constants;
}

// Function to initialize orbital parameters
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> initialize(
        int num_sats, double a_low, double a_high, double active_fraction) {
    std::vector<std::vector<double>> sat_parameters(num_sats, std::vector<double>(7));
    std::vector<std::vector<double>> sat_constants(num_sats, std::vector<double>(6));

    double AMIN = a_low;
    double AMAX = a_high;
    std::vector<int> signs = { -1, 1 };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int satNr = 0; satNr < num_sats; ++satNr) {
        double ee = dist(gen) * (EMAX - EMIN) + EMIN;
        double aa = dist(gen) * (AMAX - AMIN) + AMIN;

        double ii = dist(gen) * (IMAX - IMIN) + IMIN;
        double ww = dist(gen) * (WMAX - WMIN) + WMIN;
        double Om = dist(gen) * (OMAX - OMIN) + OMIN;
        int sign = signs[dist(gen) < 0.5 ? 0 : 1];

        const double CEarth = 9.91e-14;
        double TT = sign * sqrt(CEarth * pow(aa, 3));
        int active = (dist(gen) < active_fraction) ? 1 : 0;

        sat_parameters[satNr][0] = aa;
        sat_parameters[satNr][1] = ee;
        sat_parameters[satNr][2] = ii;
        sat_parameters[satNr][3] = ww;
        sat_parameters[satNr][4] = Om;
        sat_parameters[satNr][5] = TT;
        sat_parameters[satNr][6] = active;

        sat_constants[satNr] = constants(sat_parameters[satNr]);
    }

    return std::make_pair(sat_parameters, sat_constants);
}

std::vector<std::vector<double>> calculate_trig(int accuracy, char mode) {
    std::vector<double> E_1(accuracy);
    std::vector<double> E_2(accuracy);

    for (int i = 0; i < accuracy; ++i) {
        E_1[i] = i * 2 * M_PI / accuracy;
        E_2[i] = i * 2 * M_PI / accuracy;
    }

    std::vector<std::vector<double>> EE(accuracy, std::vector<double>(accuracy));

    for (int i = 0; i < accuracy; ++i) {
        for (int j = 0; j < accuracy; ++j) {
            EE[i][j] = E_1[j];
        }
    }

    std::vector<std::vector<double>> trig_E(accuracy, std::vector<double>(accuracy));

    for (int i = 0; i < accuracy; ++i) {
        for (int j = 0; j < accuracy; ++j) {
            if (mode == 's') {
                trig_E[i][j] = std::sin(EE[i][j]);
            }
            else {
                trig_E[i][j] = std::cos(EE[i][j]);
            }
        }
    }

    return trig_E;
}

double find_minimum (std::vector<double> parameters1, std::vector<double> parameters2,
                     std::vector<double> constants1, std::vector<double> constants2, int accuracy,
                     std::vector<std::vector<double>> sin, std::vector<std::vector<double>> cos) {

    double a1 = parameters1[0];
    double a2 = parameters2[0];
    double e1 = parameters1[1];
    double e2 = parameters2[1];

    double P11_1 = constants1[0];
    double P11_2 = constants2[0];
    double P12_1 = constants1[1];
    double P12_2 = constants2[1];
    double P21_1 = constants1[2];
    double P21_2 = constants2[2];
    double P22_1 = constants1[3];
    double P22_2 = constants2[3];
    double P31_1 = constants1[4];
    double P31_2 = constants2[4];
    double P32_1 = constants1[5];
    double P32_2 = constants2[5];

    std::vector<std::vector<double>> X1(accuracy, std::vector<double>(accuracy));
    std::vector<std::vector<double>> Y1(accuracy, std::vector<double>(accuracy));
    for (int i = 0; i < accuracy; ++i) {
        for (int j = 0; j < accuracy; ++j) {
            X1[i][j] = a1 * (cos[i][j] - e1);
            Y1[i][j] = a1 * std::sqrt(1 - e1 * e1) * sin[i][j];
        }
    }

    std::vector<std::vector<double>> x1(accuracy, std::vector<double>(accuracy));
    std::vector<std::vector<double>> y1(accuracy, std::vector<double>(accuracy));
    std::vector<std::vector<double>> z1(accuracy, std::vector<double>(accuracy));
    for (int i = 0; i < accuracy; ++i) {
        for (int j = 0; j < accuracy; ++j) {
            x1[i][j] = X1[i][j] * P11_1 + Y1[i][j] * P12_1;
            y1[i][j] = X1[i][j] * P21_1 + Y1[i][j] * P22_1;
            z1[i][j] = X1[i][j] * P31_1 + Y1[i][j] * P32_1;
        }
    }

    std::vector<std::vector<double>> X2(accuracy, std::vector<double>(accuracy));
    std::vector<std::vector<double>> Y2(accuracy, std::vector<double>(accuracy));
    for (int i = 0; i < accuracy; ++i) {
        for (int j = 0; j < accuracy; ++j) {
            X2[i][j] = a2 * (cos[i][j] - e2);
            Y2[i][j] = a2 * std::sqrt(1 - e2 * e2) * sin[i][j];
        }
    }

    std::vector<std::vector<double>> x2(accuracy, std::vector<double>(accuracy));
    std::vector<std::vector<double>> y2(accuracy, std::vector<double>(accuracy));
    std::vector<std::vector<double>> z2(accuracy, std::vector<double>(accuracy));
    for (int i = 0; i < accuracy; ++i) {
        for (int j = 0; j < accuracy; ++j) {
            x2[i][j] = X2[i][j] * P11_2 + Y2[i][j] * P12_2;
            y2[i][j] = X2[i][j] * P21_2 + Y2[i][j] * P22_2;
            z2[i][j] = X2[i][j] * P31_2 + Y2[i][j] * P32_2;
        }
    }

    std::vector<double> distances(std::pow(accuracy, 2));
    int ind = 0;
    for (int i = 0; i < accuracy; ++i) {
        for (int j = 0; j < accuracy; ++j) {
            double dist;
            dist = std::sqrt(std::pow(x1[i][j] - x2[i][j], 2) + std::pow(y1[i][j] - y2[i][j], 2)
                    + std::pow(z1[i][j] - z2[i][j], 2));

            distances[ind] = dist;
            ++ind;
        }
    }
    auto minimum_element = std::min_element(distances.begin(), distances.end());
    double minimum_distance = 0;
    if (minimum_element != distances.end()) {
        minimum_distance = *minimum_element;
    }
    return minimum_distance;
}

double collision_probability (std::vector<std::vector<double>> sat_parameters,
                              std::vector<std::vector<double>> sat_constants,
                              int sat1, int sat2, double sigma, int time_step, int accuracy,
                              std::vector<std::vector<double>> sin, std::vector<std::vector<double>> cos) {

    double col_prob = 0;
    double month_to_seconds = 30 * 24 * 60 * 60;

    if (sat_parameters[sat1][6] != -1 && sat_parameters[sat2][6] != -1) {
        bool active_satellite = sat_parameters[sat1][6] + sat_parameters[sat2][6];
        double synodic_period = 1 / std::abs(1 / sat_parameters[sat1][5] - 1 / sat_parameters[sat2][5]);
        int num_approaches = time_step * month_to_seconds / synodic_period;

        std::vector<double> parameters1 = sat_parameters[sat1];
        std::vector<double> parameters2 = sat_parameters[sat2];
        std::vector<double> constants1 =sat_constants[sat1];
        std::vector<double> constants2 =sat_constants[sat2];
        double min_distance = find_minimum(parameters1, parameters2, constants1, constants2, accuracy, sin, cos);

        double col_prob_per_approach;
        double factor;
        if (active_satellite == true) {
            factor = 5 * std::pow(10, -5);
        }
        else {
            factor = 5 * std::pow(10, -1);
        }
        col_prob_per_approach = factor * std::exp(-std::pow(min_distance, 2) / (2 * std::pow(sigma , 2)));
        col_prob = 1 - std::pow(1 - col_prob_per_approach, num_approaches);
    }
    return col_prob;
}
