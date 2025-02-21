import os
import logging
from src.utils.fileHandler import create_file, create_folder, create_textfile




def check_names(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def create_results_directory(config, KFoldIndex = -1, create_dir = True):
    """
    Function to create the results directory for the current run.
    """
    folderPath = os.path.join(os.path.join(config["paths"]["results"], config['general']['experiment_name']), config["general"]["trialNumber"])
    
    # if we're performing a bootstrapping experiment, we don't want to create a new folder "KFold1" each time, use this workaround which checks the number of folders in the directory already exist  
    if config['general']['dataset_amounts_experiment']:
        experimentPath = os.path.join(config["paths"]["results"], config['general']['experiment_name'])
        index = KFoldIndex - 1
        current_N_patients = config["data"]['n_training_patients_sets'][index]
        folderPath = os.path.join(experimentPath, f"n_patients_{current_N_patients}", f"seed_{config["general"]["seed"]}")

    elif (KFoldIndex != -1):
        folderPath = os.path.join(os.path.join(config["paths"]["results"], config['general']['experiment_name']), config["general"]["trialNumber"])
        folderPath = os.path.join(folderPath, f"KFold{KFoldIndex}")
    #folderPath += "\\"

    # Create folder if does not exist
    if create_dir:
        create_folder(folderPath)
        check_names(folderPath)

    logging.info(f"Directory path updated: {folderPath}")
    config['general']['resultsCurrentDirectory'] = folderPath



