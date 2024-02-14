import os
from model import simulation


def main():
    print("Main script started.")
    input_files_path = os.path.abspath(
        r"/Users/janlucal/Documents/GitHub/HubaldModel/Python/Optimized/input")
    input_files = []
    for entry in os.scandir(input_files_path):
        if entry.is_file() and entry.name.endswith(".txt"):
            file_path = entry.path
            input_files.append(file_path)
    print(f"A total of {len(input_files)} parameter set(s) will be simulated")

    for input_file in input_files:
        simulation(input_file)
    print("Main script finished.")


if __name__ == '__main__':
    main()
