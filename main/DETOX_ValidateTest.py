import logging
import os
import numpy as np
import pandas as pd
import wandb

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = r"C:\Users\r.van.der.wal\Documents\GitHub\pred_RT"#os.getcwd()
sys.path.insert(1, path_src)
print(path_src)



from src.config_presets.tools.get_config import get_config, load_config
from src.dataset.load_dataset import load_dataset, load_dataset_total, check_image_data_exists
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file, create_textfile

from src.evaluation.get_total_evaluation import get_total_evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    #path_config = r"C:\Users\r.van.der.wal\Documents\GitHub\pred_RT\src\config_presets\DETOXLung_config.yaml"
    #config = load_config(path_config)

    # Disable randomness
    #set_random_seed(config['general']['seed'])

    path_tostudy = r"Z:\Robert\Export\Export_results\2024_11_04\Trial_5"
    path_save = os.path.join(path_tostudy, "Test_Pred")
    get_total_evaluation(path_tostudy,path_save)

    path_labels = os.path.join(path_save,"labels.csv")
    path_predictions = os.path.join(path_save,"predictions.csv")

    df_labels = pd.read_csv(path_labels)
    df_pred = pd.read_csv(path_predictions)

    list_labels = list(df_labels.iloc[:,1])
    list_pred = list(df_pred.iloc[:,1])

    auc = roc_auc_score(list_labels, list_pred)
    acc = accuracy_score(list_labels, list_pred)
    f1 = f1_score(list_labels, list_pred)

    tn,fp,fn,tp = confusion_matrix(list_labels, list_pred).ravel()
    sensitivity = tp/ (tp+fn)
    specificity = tn/ (tn+fp)

    print(auc,acc,f1,sensitivity,specificity)
    array = np.array([auc,acc,f1,sensitivity,specificity],dtype=np.float32)
    np.save(os.path.join(path_save,"metricValues.npy"),array)
    print("----------------- FINISHED --------------------")
