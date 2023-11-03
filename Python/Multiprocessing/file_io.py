import os
import csv
import time
import numpy as np


def read_input_file(file_path=os.path.abspath(os.getcwd() + '/input/input_parameters.txt')):
    """
    Reads input parameters from an input file and returns them as a dictionary.

    Args:
        file_path (str): Directory of the input file. Defaults to /input/input_parameters.txt.

    Returns:
        dict: Input parameters and their respective keywords.
    """
    input_parameters = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split('=')
                value = value.strip()

                # Check if the value is a list
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1]  # Remove the square brackets
                    # Split the values by comma and strip leading/trailing spaces
                    value = [v.strip() for v in value.split(',')]

                    # Convert individual elements of the list to int or float if possible
                    converted_values = []
                    for v in value:
                        try:
                            if '.' in v:
                                converted_values.append(float(v))
                            else:
                                converted_values.append(int(v))
                        except ValueError:
                            converted_values.append(v)  # Keep the value as a string

                    value = converted_values

                else:
                    # Convert the value to the appropriate data type
                    try:
                        if '.' in value:
                            value = float(value)  # Convert to float if value contains a dot
                        else:
                            value = int(value)  # Convert to integer if value is an integer string
                    except ValueError:
                        if value.startswith('"') and value.endswith('"') or value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]  # Remove quotation marks if present

                input_parameters[key.strip()] = value

    return input_parameters


def save_arrays(arrays, outputDir):
    arrayNames = ["satParameters", "satConstants", "probabilityMatrix"]
    for ii, arr in enumerate(arrays):
        fileName = arrayNames[ii] + ".csv"
        fileDir = os.path.join(outputDir, fileName)
        np.savetxt(fileDir, arr, delimiter=',')


def read_arrays(arrayDir):
    try:
        # Load the CSV file into a numpy array
        data = np.genfromtxt(arrayDir, delimiter=',')

        return data

    except IOError as e:
        print("Error reading CSV file:", str(e))
        return None


def create_header(timestamps, input_parameters):
    """
    Creates a list containing lines to be printed into the output .csv file header.

    Args:
        filename (str): Name of the used data file.
        timestamps (list): List of starting time, ending time and elapsed time.
        input_parameters (dict): Input parameters of the neural network.

    Returns:
        tuple: A tuple containing a list and a string:
            - header (list): A list of lines to be printed into the output file header
            - sample_nzmber (str): Sample number.

    """
    header = []
    header_line = "Hubald model simulation data"
    header.append(header_line)
    header.append("")
    line = "starting_time = " + str(timestamps[0])
    header.append(line)
    line = "finishing_time = " + str(timestamps[1])
    header.append(line)
    line = "elapsed_time = " + str(timestamps[2]) + " s"
    header.append(line)
    for key in input_parameters:
        line = key + " = " + str(input_parameters[key])
        header.append(line)
    return header


def write_results_to_csv(simulationData, fileHeader, output_directory=os.path.abspath(os.getcwd() + '/output')):
    """
    Write the results into csv file.

    Args:


    Returns:
        None
    """
    currentTime = time.strftime("%a %b %d %H:%M:%S %Y")  # Get the current time in the desired format
    currentTime = currentTime.replace(" ", "_")  # Replace spaces with underscores
    currentTime = currentTime[4:]
    currentTime = currentTime.replace(":", "-")  # Replace colons with hyphens or any other desired character
    outputFileNameame = "hubald_simulation.csv"
    filepath = os.path.join(output_directory, outputFileNameame)
    transposedData = np.transpose(simulationData)

    # Create the last folder if the directory below exists
    lastFolder = os.path.dirname(filepath)
    if not os.path.exists(lastFolder):
        os.makedirs(lastFolder)

    file_mode = 'w' if os.path.isfile(filepath) else 'x'
    with open(filepath, file_mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for line in fileHeader:
            writer.writerow([line])
        writer.writerow([])
        writer.writerow(["Time", "Collisions per iteration", "Total collisions", "Total satellites",
                         "Active satellites", "Passive satellites", "Small fragments", "Large fragments",
                         "Small fragment collisions", "Large fragment collisions"])
        writer.writerows(transposedData)


def update_input_file(new_values, file_path=os.path.abspath(os.getcwd() + '/Input/input_parameters.txt')):
    """
    Updates the values in an input file based on the provided dictionary.

    Args:
        new_values (dict): Dictionary containing the values to be written into the input file.
        file_path (str): Directory of the input file.

    Returns:
        None
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip():
                key, value = line.split('=')
                key = key.strip()
                if key in new_values:
                    updated_value = new_values[key]
                    line = f"{key}={updated_value}\n"
            file.write(line)
