import time
from data_handling import plot_data
from model_simulation import hubald_model


def main():
    '''
    Main function.

    Returns:
        None.
    '''
    start = time.time()
    earthRadius = 6_370_000
    aLimits = (200_000 + earthRadius, 2_000_000 + earthRadius)
    simulationData = hubald_model(10000, 1200, 3, aLimits)
    print(f'Number of collisions: {int(simulationData[2][-1])}')
    finish = time.time()
    print(f'Process finished after: {round(finish - start, 2)}s')

    plot_data(simulationData)


if __name__ == '__main__':
    main()
