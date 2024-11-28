import os
import logging
from src.utils.fileHandler import create_file, create_folder, create_textfile




def check_names(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def create_results_directory(config, KFoldIndex = -1):
    """
    Function to create the results directory for the current run.
    """
    folderPath = os.path.join(os.path.join(config["paths"]["output"], config['general']['experiment_name']), config["general"]["trialNumber"])
    if(KFoldIndex != -1):
        folderPath = os.path.join(folderPath,f"KFold{KFoldIndex}")
    folderPath += "\\"

    # Create folder if does not exist
    create_folder(folderPath)
    check_names(folderPath)

    logging.info(f"Directory path updated: {folderPath}")
    config['general']['resultsCurrentDirectory'] = folderPath
    