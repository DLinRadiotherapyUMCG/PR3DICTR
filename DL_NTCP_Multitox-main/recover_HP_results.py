from ray import tune
from main import HP_tune_trial
import pandas as pd
import os
import config as cfg
# Load the result grid from the previous Ray Tune run

experiment_path = cfg.exp_root_dir

restored_tuner = tune.Tuner.restore(experiment_path, trainable=HP_tune_trial)
result_grid = restored_tuner.get_results()

df = result_grid.get_dataframe()
df.to_csv(os.path.join(experiment_path, "recovered_results.csv"), sep=";", index=False)