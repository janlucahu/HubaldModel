import os
import shutil
import time
from data_handling import plot_data
from hubald_model import hubald_model
from file_io import read_input_file, write_results_to_csv, create_header


def main():
    '''
    Main function.

    Returns:
        None.
    '''
    start = time.time()
    startingTime = time.asctime()
    inputFile = os.path.abspath(os.getcwd() + '/input/input_parameters.txt')
    inputParameters = read_input_file(inputFile)

    currentDir = os.getcwd()
    currentTime = time.strftime("%a %b %d %H:%M:%S %Y")  # Get the current time in the desired format
    currentTime = currentTime.replace(" ", "_")  # Replace spaces with underscores
    currentTime = currentTime[4:]
    currentTime = currentTime.replace(":", "-")  # Replace colons with hyphens or any other desired character
    saveDir = os.path.join(currentDir, os.path.abspath("output/" + currentTime))
    os.makedirs(saveDir)
    destinationFile = os.path.join(saveDir, os.path.basename(inputFile))
    shutil.copy(inputFile, destinationFile)

    arraysDir = os.path.abspath("/Users/janlucal/Documents/GitHub/HubaldModel/Python/Multiprocessing/Input/Matrices/50000/1")
    simulationData, colProbMatrix = hubald_model(inputParameters, saveDir, reuseArrays=False)
    print(f'Number of collisions: {int(simulationData[2][-1])}')
    finish = time.time()
    elapsedTime = finish - start
    print(f'Process finished after: {round(elapsedTime, 2)}s')
    endingTime = time.asctime()

    plot_data(simulationData, saveDir)
    timestamps = [startingTime, endingTime, elapsedTime]
    fileHeader = create_header(timestamps, inputParameters)
    write_results_to_csv(simulationData, fileHeader, saveDir)

    print("Process finished.")


if __name__ == '__main__':
    main()
