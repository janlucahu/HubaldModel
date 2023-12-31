import os
from file_io import read_input_file, update_input_file


input_parameters = read_input_file()

# read lists of parameters from variation file
parameter_variation_path = os.path.abspath(os.getcwd() + '/input/parameter_variation.txt')
parameter_variation = read_input_file(parameter_variation_path)
startingSatsList = parameter_variation.get("starting_sats")
sigmaList = parameter_variation.get("sigma")
launchesList = parameter_variation.get("starts_per_timestep")
deorbitsList = parameter_variation.get("deorbits_per_timestep")

new_values = {"starting_sats": 0, "sigma": 0, "starts_per_timestep": 0, "deorbits_per_timestep": 0}

for startingSats in startingSatsList:
    for sigma in sigmaList:
        for launches in launchesList:
            for deorbits in deorbitsList:
                new_values["starting_sats"] = startingSats
                new_values["sigma"] = sigma
                new_values["starts_per_timestep"] = launches
                new_values["deorbits_per_timestep"] = deorbits
                update_input_file(new_values)
                with open("ClosestDistance.py") as script:
                    exec(script.read())
