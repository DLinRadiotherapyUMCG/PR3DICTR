import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset, load_dataset_total, ValidateImageDataExists
from src.models.tools.save_model import save_model
from src.training.train_multi import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file, create_textfile
from src.dataset.ToxDataset import *

from src.evaluation.get_total_evaluation import get_total_evaluation
from sklearn.metrics import accuracy_score, roc_auc_score

if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    config = get_config('DETOXLung_config')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    path_tostudy = r"C:\Users\r.van.der.wal\Documents\check_cases\Trial_33"
    path_save = os.path.join(path_tostudy, "Test_Pred")
    get_total_evaluation(path_tostudy,path_save)

    path_labels = os.path.join(path_save,"labels.csv")
    path_predictions = os.path.join(path_save,"predictions.csv")

    df_labels = pd.read_csv(path_labels)
    df_pred = pd.read_csv(path_predictions)

    list_labels = list(df_labels.iloc[:,1])
    list_pred = list(df_pred.iloc[:,1])

    auc = roc_auc_score(list_labels, list_pred)

    print(auc)
    array = np.array([auc],dtype=np.float32)
    np.save(os.path.join(path_save,"aucValue.npy"),array)
    print("----------------- FINISHED --------------------")
