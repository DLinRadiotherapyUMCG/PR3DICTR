import os
from time import sleep
import ray
from ray import train, tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device

from ray.tune.progress_reporter import ProgressReporter
from ray.tune.experiment.trial import Trial
from ray.train.torch import TorchTrainer, get_device
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune import CLIReporter, ProgressReporter
from ray.tune.progress_reporter import TuneReporterBase
from ray.experimental.tqdm_ray import safe_print

from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
from ray.tune.logger import LoggerCallback
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity


class TrialTerminationReporter(CLIReporter):
    """
    A reporter that only prints the summary table of the most recent trials whenever a trial ends (completed or errored)
    (i.e. not every 30 seconds, like the default CLIReporter does).
    """
    def __init__(self, *args, **kwargs):
        super(TrialTerminationReporter,self).__init__(*args, **kwargs)
    #def __init__(self):
    #    super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == (Trial.TERMINATED or Trial.ERROR)])
        return self.num_terminated > old_num_terminated



class BasicTrialLogger(LoggerCallback):
    """"
    A logger that prints to the CLI when a trial starts, completes, or errors.
    These are intentionally designed to be short (otherwise there would be a *lot* of outputs)
    """
    def on_trial_start(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(f"{trial} - Starting trial.")

    def on_trial_complete(self, iteration, trials, trial, **info):
        print(f"{trial} - Trial complete.")
    
    def on_trial_error(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(f"{trial} - Trial errored. See logs.")




def trial_str_creator(trial):
    return f"{trial.trial_id}"




def stop_invald_trial(trial_id, result):
    """
    This function will kill a ray tune HP trial if the model config is invalid
    For example, if the model's convolutional layers make a feature map with a size of 1
    """
    return result['mean_val_AUC'] == -1



def report_invalid_model_trial(config):
    """
    This function will kill a ray tune HP trial if the model config is invalid
    For example, if the model's convolutional layers make a feature map with a size of 1
    """
    report = {"done" : True,
        "fold_n" : None, "epochs" : 'done', "total_params" : None,
            "train_loss" : -1, "val_loss" : -1,                     
            "mean_train_AUC" : -1, "mean_val_AUC" : -1}
    for endpoint in config.endpoint_list:
        report[f"mean_{endpoint}_train_AUC"] = -1
        report[f"mean_{endpoint}_val_AUC"] = -1
        report[f"mean_{endpoint}_val_ECE"] = -1
    train.report(report)



def rename_folders_for_parameter_tuning(cfg, new_exp_dir):
    cfg.exp_dir = new_exp_dir
    cfg.exp_src_dir = os.path.join(new_exp_dir, 'src')
    cfg.exp_models_dir = os.path.join(cfg.exp_src_dir, 'models')
    cfg.exp_optimizers_dir = os.path.join(cfg.exp_src_dir, 'optimizers')
    cfg.exp_data_preproc_dir = os.path.join(cfg.exp_src_dir, 'data_preproc')
    cfg.exp_figures_dir = os.path.join(new_exp_dir, 'figures')
    cfg.temp_model_weights_dir = os.path.join(new_exp_dir, 'tmp_model_weights')
    return cfg


