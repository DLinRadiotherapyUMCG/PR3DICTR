import logging

from src.training.k_fold_cross_validation import K_fold_cross_validation
import src.hyper_opt.Optuna_hpt as Optuna_hpt
import src.hyper_opt.WandB_functions as WandB_functions
from src.hyper_opt.hyperHandler import HyperTuning_Handler




class experimentHandler():
    """
    An object to help manage the main experiments. Checks the config to see if you want to do one K-folds run, or if you want
    to do a hyperparameter tuning experiment. Depending on this, it then executes your experiment.
    Also checks to see if you want to use Weights & Biases, and logs in if so
    """
    
    def __init__(self, config) -> None:
        # set WandB logging
        self.using_WandB = config['hyperparam_tuning']['WandB']['isEnabled']
        
        logging.info("Starting up experiment handler...")

        if (self.using_WandB):
            
            # Log into weights and biases
            WandB_functions.login(config)
            print("Weights and Biases is active")

        # set hyperparameter tuning
        self.hyperparam_tuning_mode = config['hyperparam_tuning']['optuna']['isEnabled']
        if self.hyperparam_tuning_mode:
            # Initialise optuna
            logging.info("Setting up optuna")
            self.hyperHandler = HyperTuning_Handler(config)
        else:
            # just do cross-validation
            pass


    def run_experiment(self, config):
        
        if(config['general']['testMode']):
            logging.info("WARNING: ------------ TEST MODE IS ACTIVE -------------")

        # if you want to do hyperparameter tuning, then run that
        if self.hyperparam_tuning_mode:
            logging.info("Use optuna for run optimalization")
            self.hyperHandler.run_optimize_experiment()
        # else, just do cross-validation
        else:
            results = K_fold_cross_validation(config)

