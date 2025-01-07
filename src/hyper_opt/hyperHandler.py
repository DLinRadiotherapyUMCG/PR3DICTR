from src.utils.set_random_seed import set_random_seed, generate_random_seed
from src.training.k_fold_cross_validation import K_fold_cross_validation
import src.hyper_opt.Optuna_hpt as Optuna_hpt

import logging

class HyperTuning_Handler():
    """
    An object to run and manage hyperparameter tuning experiments.
    """
    def __init__(self,config):
        # Set optuna
        self.Optuna_study = Optuna_hpt.Optuna_initialise_study(config) 
        self.config = config


    def run_optimize_experiment(self):
        """
        Run the hyperparameter tuning experiment
        """
        
        n_trials = self.config['hyperparam_tuning']['optuna']['n_trials']

        self.Optuna_study.optimize(lambda trial: self.run_trial(self.config, trial), n_trials)



    def run_trial(self, config, trial):
        """
        Manages a single trial of the hyperparameter tuning experiment. 
        Updates the config with the new hyperparameters, runs the K-fold cross validation and reports the results to Optuna.
        Args:
            config (dict): the configuration of the experiment
            trial (optuna.trial.Trial): the trial object of the current trial
        Returns:
            list: the results of the trial
        """
        # set a new trial number (also used as the 'group' for WandB later)
        config['general']['trialNumber'] = f"Trial_{trial.number}" 

        # set random seed
        config['general']['seed'] = generate_random_seed()
        set_random_seed(config['general']['seed'])

        # update the config of this trial
        config = self.update_trial_hyperparameters(config, trial)

        # K-fold cross validation here
        try: # it is possible that the trial fails, likely due to bad hyperparameters leading to a model that is too large
            trial_hyper_param_dict = trial.params
            all_results = K_fold_cross_validation(config, config_for_wandb=trial_hyper_param_dict)

            # aggregate the results of the K-folds, report them to Optuna
            optuna_results = self.results_handler(config, all_results)
        
        except Exception as e:
            logging.error(f"Error in trial {config['general']['trialNumber']}")
            logging.error(e)

            optuna_results = [0.0 for _ in config['hyperparam_tuning']['optuna']['objectives']]

        return optuna_results


    def update_trial_hyperparameters(self, config : dict, trial) -> dict:
        """
        Updates the config with the new hyperparameters for the current trial
        Args:
            config (dict): the configuration of the experiment
            trial (optuna.trial.Trial): the trial object of the current trial
        Returns:
            updated_config (dict): the updated configuration, with trial parameters overwriting the default ones
        """

        # Alter normal hyperparameters
        updated_config = Optuna_hpt.normal_hyperparameters(config, trial)

        # Alter any hyperparameters that are dependend on other parameters 
        updated_config = Optuna_hpt.derived_hyperparameters(updated_config, trial)

        return updated_config


    def results_handler(self, config : dict, results : dict) -> list:
        """
        reports the desired metric/loss values to optimise to Optuna
        The K-fold cross validation results contain a lot of info, this function extracts the ones we want to optimise.
        Args:
            config (dict): the configuration of the experiment
            results (dict): the results of the K-fold cross validation
        Returns:
            trial_result (list): the results of the trial
        """

        # which obkectives we want to optimise
        objectives = config['hyperparam_tuning']['optuna']['objectives'] 
        # the keys in the resuls dict (which comes from the K-fold cross validation function)
        results_keys = list(results.keys())

        # check to see that all optuna objectives exist in the results dict
        for metric in objectives:
            if metric not in results_keys:
                raise Exception(f"Metric {metric} is not in the results dictionary")
            
        # return a list of result values
        trial_result = [results[metric] for metric in objectives]

        return trial_result




    

