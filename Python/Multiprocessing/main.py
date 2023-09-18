import time
import concurrent.futures
from calculations import *
from collision_probability import calc_collision_probability

def main():
    aLimits = [200_000, 2_000_000]
    activeFraction = 0.3

    satPrameters, satConstants = initialize(1000, aLimits, activeFraction)

    max_workers = 4  # Specify the maximum number of concurrent processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = (
            executor.submit(
                calc_collision_probability,
                satPrameters[sat1],
                satPrameters[sat2],
                satConstants[sat1],
                satConstants[sat2],
                2000, 3, 20, 2
            )
            for sat1 in range(satPrameters.shape[0])
            for sat2 in range(sat1)
        )

        print("Tasks submitted")

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None and result > 10 ** (-10):
                print(result)

        print("Finished")


if __name__ == '__main__':
    start = time.time()
    main()
    finish = time.time()
    print(f"Process finished after {finish - start}s")
