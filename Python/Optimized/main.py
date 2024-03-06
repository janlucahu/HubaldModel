import os
import logging
from model import simulation, configure_logging


def main():
    log_dir = os.path.join(os.getcwd(), "output")
    configure_logging(log_dir)
    logger = logging.getLogger(__name__)
    logger.info("Main script started.")

    try:
        input_files_path = os.path.join(os.getcwd(), "batch2")
        input_files = []
        for entry in os.scandir(input_files_path):
            if entry.is_file() and entry.name.endswith(".txt"):
                file_path = entry.path
                input_files.append(file_path)
        logging.info(f"A total of {len(input_files)} parameter set(s) will be simulated")

        for input_file in input_files:
            simulation(input_file)
    except Exception as e:
        logger.exception(f"An error occurred in the main script: {e}")
    logging.info("Main script finished.")


if __name__ == '__main__':
    main()
