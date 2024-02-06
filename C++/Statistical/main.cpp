#include <iostream>
#include <vector>

#include "statistical_hubald_model.h"

int main() {
    int num_sats = 100;
    double earth_radius = 6370000;
    double a_low = earth_radius + 200000;
    double a_high = earth_radius + 2000000;
    double active_fraction = 0.5;

    auto result = initialize(num_sats, a_low, a_high, active_fraction);
    std::vector<std::vector<double>> sat_parameters = result.first;
    std::vector<std::vector<double>> sat_constants = result.second;

    std::vector<double> parameters1 = sat_parameters[0];
    std::vector<double> parameters2 = sat_parameters[1];
    std::vector<double> constants1 = sat_constants[0];
    std::vector<double> constants2 = sat_constants[1];

    int accuracy = 20;
    std::vector<std::vector<double>> sin = calculate_trig(accuracy, 's');
    std::vector<std::vector<double>> cos = calculate_trig(accuracy, 'c');

    double sigma = 2000;
    int time_step = 3;
    double col_prob;
    for (int i = 0; i < num_sats; ++i) {
        for (int j = 0; j < i; ++j) {
            int sat1 = i;
            int sat2 = j;
            col_prob = collision_probability(sat_parameters, sat_constants, sat1, sat2, sigma, time_step, accuracy,
                                             sin, cos);
            if (col_prob > 0) {
                std::cout << col_prob << std::endl;
            }
        }
    }
    return 0;
}
