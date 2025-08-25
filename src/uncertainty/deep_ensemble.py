import os
import logging
import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm 
from itertools import chain

from src.constants import DEVICE

from src.models.tools.get_classification_model import get_classification_model
from src.uncertainty.utils.dataset import get_dataset_for_uncertainty_experiments
from src.utils.set_random_seed import set_random_seed
from src.utils.loss_func.get_loss_function import get_loss_function
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.utils.saving.create_results_directory import create_results_directory
from src.config_presets.tools.save_config import save_config
from src.models.tools.get_classification_model import get_classification_model
from src.training.train import train
from src.training.validate import validate
from src.hyper_opt.WandB_functions import initialise_WandB_group, login, stop_WandB_trial
from src.utils.saving.saving_predictions import concatenate_predictions, save_predictions
from src.evaluation.total_evaluation import total_evaluation_current_fold
from src.evaluation.get_visualisations import get_visualizations

from src.uncertainty.utils.prediction_files import make_mean_predictions_dataframe
from src.dataset.LabelTypesManager import LabelTypesManager




def train_deep_ensemble_models(config):
    """
    Train a deep ensemble model.
    """
    
    set_random_seed(config['general']['seed'])  # Set the random seed for reproducibility

    """
    Make the dataset and dataloader for uncertainty experiments.
    """
    labelManager = LabelTypesManager(config=config)  # get the label types manager from the config
    
    # Load the dataset and dataloader
    train_loaders, val_loader, test_loader, metadata = get_dataset_for_uncertainty_experiments(config)
    
    # get the loss function and metric handler
    loss_function = get_loss_function(config, labelManager)
    metricHandler = mainMetricHandler(config)  # class to compute the metrics per epoch

    """
    Train each model in the ensemble. 
    Each model will be trained with a different random seed, but the same training & validation data.
    The results are be saved in separate folders for each model.
    """

    n_models = config['uncertainty']['deep_ensemble']['n_models']
    # generate a random seed for each model
    seeds = [np.random.randint(0, 10000) for _ in range(n_models)]

    print(" LEN TRAIN LOADERS", len(train_loaders))

    for train_loader_idx, train_loader in enumerate(train_loaders, start=1):

        for model_idx in range(1, n_models + 1):
            set_random_seed(seeds[model_idx - 1])  # Set the random seed for each model

            # set the directory for this model
            KFoldIndex = train_loader_idx if config['general']['dataset_amounts_experiment'] else -1
            create_results_directory(config, KFoldIndex=KFoldIndex,  folder_name=f"model_{model_idx}", UQ_experiment=True)

            logging.info(f'Fold {model_idx}/{n_models}')
            initialise_WandB_group(config, project_name=config['general']['experiment_name'], groupName=config['general']['trialNumber'], config_for_wandb=None)

            # save the config file for this model
            save_config(config)

            logging.info('Getting model')
            model = get_classification_model(config, metadata=metadata, save_summary=True)
            model.to(device=DEVICE)
            model = train(config, model, loss_function, train_loader, val_loader, metricHandler)

            stop_WandB_trial(config)
            """
            Validate the model on all sets (train, val, test) and save the results and predictions.
            """
            logging.info('Validating model')

            _, _, _, _, train_preds_dict, train_targets_dict, train_patientIDs_list = validate(config, model, loss_function, train_loader, metricHandler)
            _, _, _, _, val_preds_dict,   val_targets_dict,   val_patientIDs_list   = validate(config, model, loss_function, val_loader, metricHandler)
            _, _, _, _, test_preds_dict,  test_targets_dict,  test_patientIDs_list  = validate(config, model, loss_function, test_loader, metricHandler)
            
            # save predictions
            all_patientIDs_list = train_patientIDs_list + val_patientIDs_list + test_patientIDs_list                 # IDs column
            mode_list = ['train']*len(train_patientIDs_list) + ['val']*len(val_patientIDs_list) + ['test']*len(test_patientIDs_list)  # Mode column
            
            all_preds_dict, all_targets_dict = concatenate_predictions(config, [train_preds_dict, val_preds_dict, test_preds_dict],  # concat the predictions and labels
                                                                    [train_targets_dict, val_targets_dict, test_targets_dict])

            # save all the predictions into one csv file
            save_predictions(config, labelManager, all_patientIDs_list, all_preds_dict, all_targets_dict, mode_list)

            # collect all metrics for this fold
            sets_to_evaluate = ['train', 'val', 'test']
            total_evaluation_current_fold(config, sets=sets_to_evaluate, external_set=False)

            # make the visualisations (plots) for this fold
            get_visualizations(config, sets=sets_to_evaluate, pred_csv_dir=None, external_set=False)

            if torch.cuda.is_available():
                torch.cuda.empty_cache() # clear the GPU memory
            gc.collect()  # clear the CPU memory



        logging.info('Deep ensemble training complete!')

        logging.info('Evaluating deep ensemble models')
        experiment_dir = config['general']['resultsCurrentDirectory']
        print("ONE", experiment_dir)
        experiment_dir = os.path.dirname(experiment_dir)
        print("TWO", experiment_dir)
        evaluate_deep_ensemble_models(config, experiment_dir)



def post_hoc_evaluate_deep_ensemble_models(config, experiment_dir):
    pass

def evaluate_deep_ensemble_models(config, experiment_dir):
    """
    Evaluate the deep ensemble models.
    """
    #logging.info('Evaluating deep ensemble models')
    #experiment_dir = config['general']['resultsCurrentDirectory'] # os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    #print("EXPERIMENT DIR", experiment_dir)
    
    print("EXPERIMENT DIR", experiment_dir)
    print(os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber']))
    endpoint_list = config['columns']['labels']  # the endpoints to evaluate

    prediction_columns = [x+'_pred' for x in endpoint_list]  # the columns in the predictions csv file
    #true_label_columns = [x+'_true' for x in endpoint_list]  # the columns in the predictions csv file
    all_label_column_names = config['saving']['label_column_names']
    label_column_names = list(chain.from_iterable(
        [list(x) if isinstance(x, tuple) else [x] for x in all_label_column_names]
    ))
    true_label_columns = ['{}_true'.format(x) for x in label_column_names]

    # join each model's test set predictions into one dataframe
    #df_all_test_preds = pd.DataFrame()
    
    for model_idx in tqdm(range(1, config['uncertainty']['deep_ensemble']['n_models'] + 1)):
        #model_dir = os.path.join(config['general']['resultsCurrentDirectory'], f"model_{model_idx}")
        model_predictions_csv_dir = os.path.join(experiment_dir, f"model_{model_idx}", config['saving']['filenames']['predictions_csv'])
        df_model_preds = pd.read_csv(model_predictions_csv_dir, sep=';', index_col='PatientID')
        
        # keep only the test patients
        df_model_test_preds = df_model_preds[df_model_preds['Mode'] == 'test']
        # drop the 'Mode' column
        df_model_test_preds = df_model_test_preds.drop(columns=['Mode'])

        # sort the index to ensure the order is the same for all models
        df_model_test_preds = df_model_test_preds.sort_index()
        
        if model_idx == 1:
            length_df_test = len(df_model_test_preds)
            # change the order of the columns to have the true labels first, then the predictions
            df_model_test_preds = df_model_test_preds[[*true_label_columns, *prediction_columns]]

            # change the name of the predictions columns to include the model index
            df_model_test_preds.columns = [f"{col}_model_{model_idx}" if col in prediction_columns else col for col in df_model_test_preds.columns]
            # add the model's predictions to the dataframe
            df_all_test_preds = df_model_test_preds.copy()

        else:
            assert length_df_test == len(df_model_test_preds), "The number of test patients in each model's predictions should be the same."
            assert df_all_test_preds.index.equals(df_model_test_preds.index), "The index of the test predictions should be the same for all models."
            
            # remove the true labels from the dataframe
            df_model_test_preds = df_model_test_preds.drop(columns=true_label_columns)

            # change the name of the predictions columns to include the model index
            df_model_test_preds.columns = [f"{col}_model_{model_idx}" for col in df_model_test_preds.columns if col in prediction_columns]
            
            # merge the model's predictions with the all predictions dataframe
            df_all_test_preds = pd.merge(df_all_test_preds, df_model_test_preds, left_index=True, right_index=True, how='outer')


    all_prediction_columns = [col for col in df_all_test_preds.columns if "model_" in col]
    all_prediction_columns.sort()
    # reorder the columns to have the true labels first, then the predictions
    df_all_test_preds = df_all_test_preds[true_label_columns + all_prediction_columns]

    # save the predictions to a csv file
    df_all_test_preds.to_csv(os.path.join(experiment_dir, config['uncertainty']['filenames']['all_predictions_filename'] ), sep=';', index=True)

    """
    Save the mean predictions for each toxicity endpoint.
    This will create a new dataframe with the true labels and the mean predictions for each endpoint.
    """
    # make a file with the true labels and mean predictions for each toxicity
    df_mean_predictions = make_mean_predictions_dataframe(df_all_test_preds, endpoint_list)
    
    df_mean_predictions.to_csv(os.path.join(experiment_dir, config['uncertainty']['filenames']['mean_predictions_filename'] ), sep=';', index=True)


    logging.info('Finished collecting deep ensemble model predictions!')
    
    