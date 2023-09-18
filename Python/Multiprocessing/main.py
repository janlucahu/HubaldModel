import time
import concurrent.futures
from calculations import *
from collision_probability import calc_collision_probability


def main():
    aLimits = [200_000, 2_000_000]
    activeFraction = 0.3

    satPrameters, satConstants = initialize(1000, aLimits, activeFraction)

    probabilities = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sat1 in range(satPrameters.shape[0]):
            for sat2 in range(sat1):
                probabilities.append(
                    executor.submit(calc_collision_probability, satPrameters[sat1], satPrameters[sat2], satConstants[sat1],
                                    satConstants[sat2], 2000, 3, 20, 2)
                )

    for prob in concurrent.futures.as_completed(probabilities):
        if prob.result() > 10 ** (-10):
            print(prob.result())


if __name__ == '__main__':
    start = time.time()
    main()
    finish = time.time()
    print(f"Process finished after {finish - start}s")
