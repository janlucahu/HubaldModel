#ifndef STATISTICAL_STATISTICAL_HUBALD_MODEL_H
#define STATISTICAL_STATISTICAL_HUBALD_MODEL_H

#include <vector>
#include <utility>

std::vector<double> constants(const std::vector<double>& sat_parameters);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> initialize(
        int num_sats, double a_low, double a_high, double active_fraction);

std::vector<std::vector<double>> calculate_trig(int accuracy, char mode);

double find_minimum (std::vector<double> parameters1, std::vector<double> parameters2,
                     std::vector<double> constants1, std::vector<double> constants2, int accuracy,
                     std::vector<std::vector<double>> sin, std::vector<std::vector<double>> cos);

double collision_probability(std::vector<std::vector<double>> sat_parameters,
                             std::vector<std::vector<double>> sat_constants,
                             int sat1, int sat2, double sigma, int time_step, int accuracy,
                             std::vector<std::vector<double>> sin, std::vector<std::vector<double>> cos);


#endif //STATISTICAL_STATISTICAL_HUBALD_MODEL_H
