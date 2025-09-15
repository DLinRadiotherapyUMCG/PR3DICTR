import os
import optuna
import logging

from src.utils.fileHandler import create_file, create_folder

# TODO: comments and documentation for all of these function


def update_config(dic, location, suggested_value):
    """
    Function to update the config based on a given location and 
    """
    if len(location) == 1:
        dic[location[0]] = suggested_value
        return dic
    else:
        dic[location[0]] = update_config(dic[location[0]], location[1:], suggested_value)
        return dic

def normal_hyperparameters(config, trial):
    for key in config['hyperparam_tuning']['hyperparams'].keys():
        try:
            hyperinfo = config['hyperparam_tuning']['hyperparams'][key]
            location = hyperinfo['location']

            # Check if multi is encountered
            if(hyperinfo['name'].startswith("n_")):
                suggested_value = generate_value(trial, hyperinfo, config['general']['firstRun'])
            else:   
                suggested_value = generate_value(trial, hyperinfo)
            
            config = update_config(config,location,suggested_value)
        except Exception as error:
                print(f"Could not generate hyperparams [Single] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
    return config



def derived_hyperparameters(config, trial):
    '''
    Some hyperparameters are connected/ interdependend, here derived 
    hyperparameters are to be set.
    
    '''
    # If the number of down blocks is specified there should be an equal amount of related parameters          
    if 'n_clinical_layers' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_clinical_layers']:
            try:
                temp_list = []
                hyperinfo = config['hyperparam_tuning']['derived']['n_clinical_layers'][key]  
                location = hyperinfo['location']
                base_name = hyperinfo['name']

                for i in range(config['model']['output_head']['n_clinical_layers']):
                    hyperinfo_new = hyperinfo.copy()
                    hyperinfo_new['name'] = base_name + str(i)
                    suggested_value = generate_value(trial, hyperinfo_new)
                    temp_list.append(suggested_value)
                config = update_config(config,location,temp_list)

            except Exception as error:
                print(f"Could not generate hyperparams [Derived] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
            
    if 'n_linear_layers' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_linear_layers']:
            try:
                temp_list = []
                hyperinfo = config['hyperparam_tuning']['derived']['n_linear_layers'][key]  
                location = hyperinfo['location']
                base_name = hyperinfo['name']
                if "clinical_variables_position" in base_name.lower():  # which layer to add the clinical variables (is dependent on the number of shared linear layers)
                    hyperinfo['max'] = config['model']['output_head']['n_linear_layers'] + 1 # the max should be the index of the last shared linear layer + 1
                    suggested_value = generate_value(trial, hyperinfo)
                    config = update_config(config,location,suggested_value)
                else:
                    for i in range(config['model']['output_head']['n_linear_layers']):
                        hyperinfo_new = hyperinfo.copy()
                        hyperinfo_new['name'] = base_name + str(i)
                        suggested_value = generate_value(trial, hyperinfo_new)
                        temp_list.append(suggested_value)
                    config = update_config(config,location,temp_list)
            except Exception as error:
                print(f"Could not generate hyperparams [Derived] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
            
    if 'n_linear_layers_endpoint' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_linear_layers_endpoint']:
            try:
                temp_list = []
                hyperinfo = config['hyperparam_tuning']['derived']['n_linear_layers_endpoint'][key]  
                location = hyperinfo['location']
                base_name = hyperinfo['name']
                for i in range(config['model']['output_head']['n_linear_layers_endpoint']):
                    hyperinfo_new = hyperinfo.copy()
                    hyperinfo_new['name'] = base_name + str(i)
                    suggested_value = generate_value(trial, hyperinfo_new)
                    temp_list.append(suggested_value)
                config = update_config(config,location,temp_list)
            except Exception as error:
                print(f"Could not generate hyperparams [Derived] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
        
    return config

# def set_min_value(hyperinfo, firstRun):
#     minValue = hyperinfo['min']
#     if firstRun:
#         minValue = hyperinfo['max']
#     return minValue


def generate_value(trial, hyperinfo, firstRun = False):
    """
    Selects and generates a value based on hyperinfo
    """
    keys = hyperinfo.keys()
    step = None
    log = False
    if 'step' in keys:
        step = hyperinfo['step']
    if 'log' in keys:
        log = hyperinfo['log']

    if hyperinfo['type'] == 'int':
        if step == None:
            step = 1
        suggested_value = trial.suggest_int(hyperinfo['name'], hyperinfo['min'], hyperinfo['max'], step=step, log=log)
    if hyperinfo['type'] == 'float':
        suggested_value = trial.suggest_float(hyperinfo['name'], hyperinfo['min'], hyperinfo['max'], step=step, log=log)
    if hyperinfo['type'] == 'categorical':
        suggested_value = trial.suggest_categorical(hyperinfo['name'], hyperinfo['options'])       
    if hyperinfo['type'] == 'dis_un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'], hyperinfo['min'], hyperinfo['max'], hyperinfo['q'])       
    if hyperinfo['type'] == 'log_un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'], hyperinfo['min'], hyperinfo['max'])
    if hyperinfo['type'] == 'un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'], hyperinfo['min'], hyperinfo['max'])       
    
    return suggested_value




def initialize_optuna_study(config):
    study = None
    if config['hyperparam_tuning']['optuna']['isEnabled']:

        # Check where to keep study information
        studyName = config['general']['experiment_name']
        pathOptunaStudyTracker = os.path.join(os.path.join(config['paths']['results'] , studyName), "track_optuna.db")
        storage_name = f"sqlite:///{pathOptunaStudyTracker}"
        create_file(pathOptunaStudyTracker)
        storage = optuna.storages.RDBStorage(
            url = storage_name,
            engine_kwargs = {
                'pool_size' : 20,
                'max_overflow' : 0
            }
        )
        logging.info(f"Using Optuna storage at: {storage_name}")

        sampler = optuna.samplers.TPESampler(
                              n_startup_trials = config['hyperparam_tuning']['optuna']['n_startup_trials'],  # how many random trials to run before starting the bayesian optimization
                              #seed = config['general']['seed'],             # NOTE: this seed is not used for the random state of the sampler, prevents multiple processes from suggesting the same values       
                              multivariate = True,
                              constant_liar = True,                # NOTE: this parameter helps to prevent the sampler from suggesting very similar values each time. This is beneficial if model training is very costly/takes long
                            )

        study = optuna.create_study(
                            study_name = studyName, 
                            storage = storage, 
                            sampler = sampler,
                            load_if_exists = True,
                            directions = config['hyperparam_tuning']['optuna']['objective_direction'],
                            )

        # check if an experiment has been run before
        config['general']['firstRun'] = (len(study.get_trials()) == 0)
        if config['general']['firstRun']:
            logging.info("First optuna run has been detected!")
        else:
            logging.info("Continuing existing optuna run!")

    return study



