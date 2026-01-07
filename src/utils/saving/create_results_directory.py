import os
import logging
from src.utils.fileHandler import create_folder

def check_names(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def create_results_directory(config, KFoldIndex = -1, create_dir = True, folder_name = None, UQ_experiment=False):
    """
    Function to create the results directory for the current run.
    """
    folderPath = os.path.join(os.path.join(config["paths"]["results"], config['general']['experiment_name']), config["general"]["trialNumber"])
    
    # if we're performing a bootstrapping experiment, we don't want to create a new folder "KFold1" each time, use this workaround which checks the number of folders in the directory already exist  
    if config['general']['dataset_amounts_experiment']:
        
        index = KFoldIndex - 1
        current_N_patients = config["data"]['n_training_patients_list'][index]

        if UQ_experiment:
            experimentPath = os.path.join(os.path.join(config["paths"]["results"], config['general']['experiment_name']), config["general"]["trialNumber"])
        else:
            experimentPath = os.path.join(config["paths"]["results"], config['general']['experiment_name'])
        
            config["general"]["trialNumber"] = f"n_patients_{current_N_patients}"   # set the trial folder to be the number of patients
        
        seed = config["general"]["seed"]

        if folder_name is None:
            folder_name = f"seed_{seed}"
        folderPath = os.path.join(experimentPath, f"n_patients_{current_N_patients}", folder_name)

    elif folder_name is not None:
        folderPath = os.path.join(os.path.join(config["paths"]["results"], config['general']['experiment_name']), config["general"]["trialNumber"])
        folderPath = os.path.join(folderPath, folder_name)

    elif (KFoldIndex != -1):
        folderPath = os.path.join(os.path.join(config["paths"]["results"], config['general']['experiment_name']), config["general"]["trialNumber"])
        folderPath = os.path.join(folderPath, f"KFold{KFoldIndex}")

    # Create folder if does not exist
    if create_dir:
        create_folder(folderPath)
        check_names(folderPath)

    logging.info(f"Directory path updated: {folderPath}")
    config['general']['resultsCurrentDirectory'] = folderPath



