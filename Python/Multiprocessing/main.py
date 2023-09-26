import time
import multiprocessing
import numpy as np
from calculations import initialize
from collision_probability import calc_collision_probability


def main():
    aLimits = [200_000, 2_000_000]
    activeFraction = 0.3
    satParameters, satConstants = initialize(20, aLimits, activeFraction)
    sigma = 2000
    timestep = 3
    acc = 20

    E_1 = np.linspace(0, 2 * np.pi, acc)
    E_2 = np.linspace(0, 2 * np.pi, acc)

    E1, E2 = np.meshgrid(E_1, E_2)
    sinE, cosE = np.sin(E1), np.cos(E1)

    processes = []
    for sat1 in range(satParameters.shape[0]):
        for sat2 in range(sat1):
            p = multiprocessing.Process(target=calc_collision_probability, args=[satParameters[sat1], satParameters[sat2],
                                                                                 satConstants[sat1], satConstants[sat2],
                                                                                 sigma, timestep, sinE, cosE])
            p.start()
            processes.append(p)

    for process in processes:
        process.join()


if __name__ == '__main__':
    start = time.time()
    main()
    finish = time.time()
    print(f"Finalized after {finish - start}s")
