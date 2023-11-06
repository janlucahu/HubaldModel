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
fragColProbList = parameter_variation.get("fragment_collision_prob")

new_values = {"starting_sats": 0, "sigma": 0, "starts_per_timestep": 0, "deorbits_per_timestep": 0,
              "fragment_collision_prob": 0}

for startingSats in startingSatsList:
    for sigma in sigmaList:
        for launches in launchesList:
            for deorbits in deorbitsList:
                for colProb in fragColProbList:
                    new_values["starting_sats"] = startingSats
                    new_values["sigma"] = sigma
                    new_values["starts_per_timestep"] = launches
                    new_values["deorbits_per_timestep"] = deorbits
                    new_values["fragment_collision_prob"] = colProb
                    update_input_file(new_values)
                    with open("main.py") as script:
                        exec(script.read())
