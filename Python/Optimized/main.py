import os
from model import simulation


def main():
    input_files_path = os.path.abspath(r"C:\Users\jlhub\Documents\Studium\Masterarbeit\HubaldModell\HubaldModel\Python\input_files")

    input_files = []
    for entry in os.scandir(input_files_path):
        if entry.is_file() and entry.name.endswith(".txt"):
            file_path = entry.path
            input_files.append(file_path)

    for input_file in input_files:
        simulation(input_file)


if __name__ == '__main__':
    main()
