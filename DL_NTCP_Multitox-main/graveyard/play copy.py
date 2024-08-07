#!/usr/bin/env python
import config as cfg
import argparse
import time
import config
import tempfile
import os
import ray
from ray import train, tune
from ray.tune.logger import LoggerCallback

from main import perform_N_fold_cross_validation, initialize

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


    #print("Best hyperparameters: ", results.get_best_result().config)