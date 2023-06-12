import numpy as np
import multiprocessing as mp
import time

EMIN, EMAX = 0, 0.9
IMIN, IMAX = 0, np.pi
WMIN, WMAX = 0, 2 * np.pi
OMIN, OMAX = 0, 2 * np.pi
MMIN, MMAX = 0, 1000


def initialize(nrOfSats, size, accuracy=1, plane=True):
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



def compute_position(satParameter, absoluteTime, satNr):
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

    return [xx, yy, zz]


def orbital_position(satParameter, absoluteTime):
    num_satellites = satParameter.shape[0]
    satPosition = np.empty((num_satellites, 3))

    with mp.Pool() as pool:
        results = [pool.apply_async(compute_position, args=(satParameter, absoluteTime, i)) for i in range(num_satellites)]
        for i, result in enumerate(results):
            satPosition[i] = result.get()

    return satPosition


if __name__ == '__main__':
    par = initialize(100000, 1000000)

    start = time.time()
    orbital_position(par, 1000)
    finish = time.time()

    print(finish - start)
