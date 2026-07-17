import optuna
import logging
import traceback
import os
import copy

from src.utils.set_random_seed import set_random_seed, generate_random_seed
from src.training.k_fold_cross_validation import K_fold_cross_validation
from src.hyper_opt.exceptions import EarlyStopTrial
import src.hyper_opt.utils as utils

class OptunaExperimentManager():
    """
    An object to run and manage hyperparameter tuning experiments.
    """
    def __init__(self,config):
        # Set optuna
        self.Optuna_study = utils.initialize_optuna_study(config) 
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
        if config['hyperparam_tuning']['optuna']['set_new_seeds']:
            config['general']['seed'] = generate_random_seed()
            set_random_seed(config['general']['seed'])

        # update the config of this trial
        trial_config = copy.deepcopy(config)
        trial_config = self.update_trial_hyperparameters(trial_config, trial)

        # K-fold cross validation here
        try: # it is possible that the trial fails, likely due to bad hyperparameters leading to a model that is too large
            trial_hyper_param_dict = trial.params
            prune_trial_callback = lambda fold_idx, mean_val_metric_value: self.should_prune_trial(trial, fold_idx, mean_val_metric_value)
            all_results = K_fold_cross_validation(trial_config, config_for_wandb=trial_hyper_param_dict, prune_trial_callback=prune_trial_callback)
    
            # aggregate the results of the K-folds, report them to Optuna
            optuna_results = self.results_handler(trial_config, all_results)

        except EarlyStopTrial:
            # the trial was intentionally stopped early by our own pruning rule (see should_prune_trial())
            # this is not a failure, so let Optuna handle it natively (marks the trial as PRUNED, not FAILED)
            logging.info(f"Trial {trial_config['general']['trialNumber']} was pruned early.")
            raise optuna.exceptions.TrialPruned()

        except Exception as e:
            # print the trial's error to the logger (i.e. the print statements)
            logging.error(f"Error in trial {trial_config['general']['trialNumber']}. See error log .txt file for details.")
            logging.error(e)

            optuna_results = [0.0 for _ in trial_config['hyperparam_tuning']['optuna']['objectives']]

            # also save the error to a text file (mostly for debugging later)
            trial_dir = os.path.join(trial_config["paths"]["results"], trial_config['general']['experiment_name'], trial_config["general"]["trialNumber"])
            with open(os.path.join(trial_dir, 'error_log.txt'), 'a') as f:
                f.write(str(e) + "\n")
                f.write(traceback.format_exc())

            # tell optuna to ignore this failed trial
            trial.set_user_attr("failed", True)

            raise optuna.exceptions.TrialPruned()  # NOTE: this will set the trial to pruned
            
        return optuna_results

    def should_prune_trial(self, trial, fold_idx, mean_val_metric_value) -> bool:
        """
        Rule to decide whether a trial should be stopped early, based on its intermediate (per-fold) results.
        This intentionally does NOT use Optuna's built-in pruners / trial.should_prune() - it is a simple,
        fully-owned rule that can be tuned/replaced independently of Optuna's pruning machinery.

        The rule: after a configurable number of warmup folds, prune the trial if its mean validation metric
        so far is worse than a fixed threshold defined in the config (`kill_trial_threshold`).
        Args:
            trial (optuna.trial.Trial): the current trial
            fold_idx (int): the fold number just completed (1-indexed)
            mean_val_metric_value (float): this trial's mean validation metric across folds completed so far
        Returns:
            bool: True if the trial should be stopped early
        """
        optuna_config = self.config['hyperparam_tuning']['optuna']
        n_warmup_folds = optuna_config.get('pruner_n_warmup_folds', 3)
        kill_trial_threshold = optuna_config['kill_trial_threshold']

        # report the intermediate value, so it's recorded on the trial (useful for inspection/plotting later)
        trial.report(mean_val_metric_value, step=fold_idx)

        if fold_idx < n_warmup_folds:
            return False  # always let a trial complete a minimum number of folds before considering pruning it

        direction = optuna_config['objective_direction'][0]
        if direction == 'maximize':
            should_prune = mean_val_metric_value < kill_trial_threshold
            if should_prune:
                logging.info(f"Pruning trial {trial.number} at fold {fold_idx}. Mean validation metric {mean_val_metric_value} < threshold {kill_trial_threshold} (objective direction: {direction}).")
            return should_prune
        else:
            should_prune = mean_val_metric_value > kill_trial_threshold
            if should_prune:
                logging.info(f"Pruning trial {trial.number} at fold {fold_idx}. Mean validation metric {mean_val_metric_value} > threshold {kill_trial_threshold} (objective direction: {direction}).")
            return should_prune
        

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
        updated_config = utils.normal_hyperparameters(config, trial)

        # Alter any hyperparameters that are dependend on other parameters 
        updated_config = utils.derived_hyperparameters(updated_config, trial)

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

