/* This program simulates the Kessler Syndrome from a complex physics perspective.
 * This is an attempt to implement the Hubald Model in C++ to lower calculation times in respect to the original python
 * version */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>


using namespace std;

double roundto (double value, int decimals) {
    value = round(value * pow(10, decimals)) / pow(10, decimals);
    return value;
};

double random_double (double low, double high) {
    double randomDouble = low + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(high-low)));

    return randomDouble;
};

vector<double> orbital_position (double aa, double ee, double ii, double ww, double Om, double M0, double TT,
                                 int absoluteTime, int accuracy) {
    vector<double> position;
    double MM;
    double EE;
    double XX;
    double YY;
    double P11, P12, P21, P22, P31, P32;
    double xx;
    double yy;
    double zz;

    MM = M0 + 2 * M_PI / TT * absoluteTime;
    EE = MM + ee * sin(MM) + 1/2 * pow(ee, 2) * sin(2 * MM);

    XX = aa * (sin(EE) - ee);
    YY = aa * sqrt(1 - pow(ee,2)) * sin(EE);

    P11 = (cos(Om) * cos(ww) - sin(Om) * cos(ii) * sin(ww));
    P12 = (- cos(Om) * sin(ww) - sin(Om) * cos(ii) * cos(ww));
    P21 = (sin(Om) * cos(ww) + cos(Om) * cos(ii) * sin(ww));
    P22 = (- sin(Om) * sin(ww) + cos(Om) * cos(ii) * cos(ww));
    P31 = sin(ii) * sin(ww);
    P32 = sin(ii) * cos(ww);

    xx = XX * P11 + YY * P12;
    yy = XX * P21 + YY * P22;
    zz = XX * P31 + YY * P32;
    position.push_back(roundto(xx, accuracy));
    position.push_back(roundto(yy, accuracy));
    position.push_back(roundto(zz, accuracy));
    return position;
};

vector<vector<double>> initialize (int nrOfSats, int size, int tmax, double accuracy = 1, bool plane = true) {

    vector<vector<double>> satelliteValues;
    double a, e, i, w, O, M_0, T;
    double xx, yy ,zz;
    vector<double> positions;
    vector<double> parameters;
    vector<double> satVal = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int satNr = 0; satNr < nrOfSats; satNr++) {
        e = random_double(0, 0.9);
        a = random_double(0.1 * size, (size / 2) / (1 + e));

        if (plane) {
            i = 0;
        } else {
            i = random_double(0, 2 * M_PI);
        }

        w = random_double(0, 2 * M_PI);
        O = random_double(0, 2 * M_PI);
        M_0 = random_double(0, 1000);
        T = random_double((1 / 5) * tmax, tmax);

        parameters = {a, e, i, w, O, M_0, T};

        xx = orbital_position(a, e, i, w, O, M_0, T, 0, accuracy)[0];
        yy = orbital_position(a, e, i, w, O, M_0, T, 0, accuracy)[1];
        zz = orbital_position(a, e, i, w, O, M_0, T, 0, accuracy)[2];
        positions = {xx, yy, zz};

        copy(positions.begin(), positions.end(), satVal.begin());
        copy(parameters.begin(), parameters.end(), satVal.begin()+3);

        satelliteValues.push_back(satVal);
    }
    return satelliteValues;
};

int check_collision (vector<vector<double>> satValues) {
    int collisions = 0;
    for (int sat1 =  1; sat1 < satValues.size(); sat1++) {
        for (int sat2 = 0; sat2 < sat1; sat2++) {
            vector<double> satPos1 = {satValues[sat1].begin(), satValues[sat1].begin() + 2};
            vector<double> satPos2 = {satValues[sat2].begin(), satValues[sat2].begin() + 2};
            if (satPos1 == satPos2) {
                cout << "Collision between Sat Numbers: " << sat1 << "  " << sat2 << endl;
                cout << "sat1: " << satPos1[0] << "  " << satPos1[1] << "  " << satPos1[2] << endl;
                cout << "sat2: " << satPos2[0] << "  " << satPos2[1] << "  " << satPos2[2] << endl;
                collisions++;

            }
        }
    }
    return collisions;
};

int kessler_simulation (int nrOfSats, int size, int tmax, int acc=1, bool plane=true) {
    vector<vector<double>> satValues = initialize(nrOfSats, size, tmax, acc, plane);
    int nrOfCollisions = 0;
    double a, e, i, w, O, M_0, T;
    double xx, yy ,zz;
    vector<double> positions;
    vector<double> parameters;
    vector<double> satVal = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int sec = 0; sec < tmax; sec++) {
        cout << "Progress: " << sec << " of " << tmax << endl;
        for (int satNr = 0; satNr < satValues.size(); satNr++) {
            a = satValues[satNr][3];
            e = satValues[satNr][4];
            i = satValues[satNr][5];
            w = satValues[satNr][6];
            O = satValues[satNr][7];
            M_0 = satValues[satNr][8];
            T = satValues[satNr][9];
            vector<double> satPosition = orbital_position(a, e, i, w, O, M_0, T, sec, acc);
            copy(satPosition.begin(), satPosition.end(), satValues[satNr].begin());

        }
        int collisions = check_collision(satValues);
        for (int cols = 0; cols < collisions; cols++) {
            e = random_double(0, 0.9);
            a = random_double(0.1 * size, (size / 2) / (1 + e));

            if (plane) {
                i = 0;
            } else {
                i = random_double(0, 2 * M_PI);
            }

            w = random_double(0, 2 * M_PI);
            O = random_double(0, 2 * M_PI);
            M_0 = random_double(0, 1000);
            T = random_double((1 / 5) * tmax, tmax);

            parameters = {a, e, i, w, O, M_0, T};
            xx = orbital_position(a, e, i, w, O, M_0, T, sec, acc)[0];
            yy = orbital_position(a, e, i, w, O, M_0, T, sec, acc)[1];
            zz = orbital_position(a, e, i, w, O, M_0, T, sec, acc)[2];

            positions = {xx, yy, zz};

            copy(positions.begin(), positions.end(), satVal.begin());
            copy(parameters.begin(), parameters.end(), satVal.begin()+3);
            satValues.push_back(satVal);

            nrOfCollisions++;
        }
    }
    cout << "Nr of collisions: " << nrOfCollisions;
    return nrOfCollisions;
};

int main() {
    srand(time(NULL));
    kessler_simulation(1000, 1000, 1000, 1);
    /*
    vector<vector<double>> vec = {{0.9, -1.8, 0}, {-0.5, -0.7, 0}, {0.9, -1.8, -0}};
    int col = check_collision(vec);
    cout << col;
    */
    return 0;
}
