import time
from data_handling import plot_data
from model_simulation_sparse import hubald_model
from file_io import read_input_file, write_results_to_csv, create_header


def main():
    '''
    Main function.

    Returns:
        None.
    '''
    start = time.time()
    startingTime = time.asctime()
    inputParameters = read_input_file()
    simulationData, colProbMatrix = hubald_model(inputParameters)
    print(f'Number of collisions: {int(simulationData[2][-1])}')
    finish = time.time()
    elapsedTime = finish - start
    print(f'Process finished after: {round(elapsedTime, 2)}s')
    endingTime = time.asctime()

    plot_data(simulationData)
    timestamps = [startingTime, endingTime, elapsedTime]
    fileHeader = create_header(timestamps, inputParameters)
    write_results_to_csv(simulationData, fileHeader)


if __name__ == '__main__':
    main()
