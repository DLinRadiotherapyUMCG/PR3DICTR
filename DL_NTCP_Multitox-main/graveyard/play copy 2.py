#!/usr/bin/env python

import argparse
import time
import tempfile
import os
import ray
from ray import train, tune
from ray.tune.logger import LoggerCallback



class TestLoggerCallback(LoggerCallback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"TestLogger for trial {trial}: {result}")


def trial_str_creator(trial):
    #config.exp_dir = trial.trial_name
    return "{}_{}_123".format(trial.trainable_name, trial.trial_id)


def evaluation_fn(step, width, height):
    time.sleep(0.1)
    print("HELLO", ray.train.get_context().get_experiment_name())
    print(ray.train.get_context().get_trial_name())
    print(ray.train.get_context().get_trial_dir())
    print(ray.train.get_context().get_metadata())
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


def easy_objective(config):
    # Hyperparameters
    from main import perform_N_fold_cross_validation
    
    import config as cfg
    #cfg.root_path = "//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Daniel MacRae/DL_NTCP_Multitox"

    cfg.cv_folds = config["folds"]
    cfg.max_epochs = config["max_epochs"]
    cfg.batch_size = config["batch_size"]

    #logger = initialize(torch_version=cfg.torch_version, device=cfg.device, create_folders=False)
    mean_val_auc_all_folds = evaluation_fn(config["folds"], config["max_epochs"], config["batch_size"] )
    # Feed the score back back to Tune.
    train.report({"mean_AUC": mean_val_auc_all_folds})


if __name__ == "__main__":

    smoke_test = True

    tuner = tune.Tuner(
        easy_objective,
        run_config=train.RunConfig(
            name="CV_1",
            #callbacks=[TestLoggerCallback()],
            #stop={"training_iteration": 5 if smoke_test else 100},
        ),
        tune_config=tune.TuneConfig(
            metric="mean_AUC",
            mode="max",
            num_samples=5,
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
        ),
        param_space={
            "batch_size": tune.randint(8, 32),
            "folds": tune.randint(1, 4),
            "max_epochs": tune.grid_search([8, 16, 32])
        },
    )
    results = tuner.fit()

    #print("Best hyperparameters: ", results.get_best_result().config)