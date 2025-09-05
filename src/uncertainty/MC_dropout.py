import os
import logging
import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm 

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
from src.uncertainty.utils.collect_predictions import collect_one_predictions_pass
from src.dataset.LabelTypesManager import LabelTypesManager



def train_MC_dropout_model(config, UQ_method = "MC_dropout"):

    

    set_random_seed(config['general']['seed'])  # Set the random seed for reproducibility

    """
    Make the dataset and dataloader for uncertainty experiments.
    """

    # set dropout probability for MC dropout
    config['model']['dropout_p'] = config['uncertainty']['MC_dropout']['dropout_p'] 
    config['model']['TransRP']['vit_dropout_p'] = config['uncertainty']['MC_dropout']['dropout_p']  # set the dropout probability for the TransRP model
    
    labelManager = LabelTypesManager(config=config)  # get the label types manager from the config
    # get the loss function and metric handler
    loss_function = get_loss_function(config, labelManager)
    metricHandler = mainMetricHandler(config)  # class to compute the metrics per epoch
    # config['saving']['label_column_names'] = labelManager.label_names_full_list
    
    # Load the dataset and dataloader
    #train_loaders, val_loader, test_loader, metadata
    df_train_list, df_val, df_test = get_dataset_for_uncertainty_experiments(config)
    train_transforms, val_transforms = get_transforms(config)
    test_set_transforms = val_transforms if UQ_method != "TTA" else train_transforms

    val_loader, _ = make_dataloader(config, df_val, val_transforms, validation_mode=True)  
    test_loader, _ = make_dataloader(config, df_test, test_set_transforms, validation_mode=True)

    print(" LEN TRAIN LOADERS", len(df_train_list))
    for train_loader_idx, df_train in enumerate(df_train_list, start=1):

        
        train_loader, metadata = make_dataloader(config, df_train, train_transforms, validation_mode=False)

        print(len(df_train))
        print(metadata)
    
        # train the model
        # set the directory 
        KFoldIndex = train_loader_idx if config['general']['dataset_amounts_experiment'] else -1
        folder_name = f"model_1"
        create_results_directory(config, KFoldIndex=KFoldIndex, folder_name=folder_name, UQ_experiment=True)

        logging.info(f'Training model')
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

        experiment_dir = config['general']['resultsCurrentDirectory'] # os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'], f'model_{model_idx}')
        collect_bayesian_forward_passes(config, experiment_dir, test_loader, metadata, UQ_method = 'MC_dropout')

        del train_loader.dataset
        del train_loader
        model.to('cpu')
        del model

        # Explicitly delete any remaining references and clear CUDA cache
        for obj_name in ['train_loader', 'metadata', 'loss_function', 'metricHandler']:
            if obj_name in locals():
                del locals()[obj_name]
        torch.cuda.empty_cache()
        gc.collect()


from src.dataset.get_transforms import get_transforms
from src.dataset.load_dataset import load_dataset, generate_K_fold_cross_validation_splits
from src.dataset.get_dataloader import make_dataloader   
from src.models.tools.save_model import load_model
from src.config_presets.tools.load_config import load_config
from itertools import chain





def post_hoc_collect_bayesian_forward_passes(config, UQ_method = "MC_dropout"):
    # MAKE TEST LOADER AND STUFF HERE
    # get the test set data and make a test dataloader
    df_train_val, df_test = load_dataset(config)
    train_transforms, val_transforms = get_transforms(config)

    if UQ_method == 'MC_dropout':
        transforms_method = val_transforms
    elif UQ_method == 'TTA':
        transforms_method = train_transforms
        
    else:
        raise ValueError(f"Unrecognized UQ method: {UQ_method}. Supported methods are 'MC_dropout' and 'TTA'.")

    test_loader, metadata = make_dataloader(config, df_test, transforms_method, validation_mode=True)

    experiment_dir = config['general']['resultsCurrentDirectory'] # os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'], 'model_1')
    #model_config = load_config(os.path.join(experiment_dir, config['saving']['filenames']['config_yaml']) )

    collect_bayesian_forward_passes(config=config, experiment_dir=experiment_dir, UQ_method=UQ_method, test_loader=test_loader, metadata=metadata)





def collect_bayesian_forward_passes(config, experiment_dir, test_loader, metadata, UQ_method = 'MC_dropout',):

    # labelManager = LabelTypesManager(config=config)  # get the label types manager from the config
    #config['saving']['label_column_names'] = labelManager.label_names_full_list

    #print((experiment_dir, config['saving']['filenames']['config_yaml']) )
    model_config = load_config(os.path.join(experiment_dir, config['saving']['filenames']['config_yaml']) )

    model_config['paths']['results'] = model_config['paths']['results']
    model_config['general']['resultsCurrentDirectory'] = config['general']['resultsCurrentDirectory']

    # load in the model weights
    model = get_classification_model(model_config, metadata=metadata, save_summary=False)
    model.to(DEVICE)
    model = load_model(model_config, model) # load the saved weights
    enable_MC_dropout = True if UQ_method == 'MC_dropout' else False  # enable dropout at inference tme if the UQ method is MC dropout

    # print("dropout is enabled:", enable_MC_dropout)
    # print(model_config['model']['dropout_p'])

    # use the validate function to get the predictions
    endpoint_list = config['columns']['labels']  # the endpoints to evaluate

    prediction_columns = [x+'_pred' for x in endpoint_list]  # the columns in the predictions csv file

    all_label_column_names = config['saving']['label_column_names']
    label_column_names = list(chain.from_iterable(
        [list(x) if isinstance(x, tuple) else [x] for x in all_label_column_names]
    ))
    true_label_columns = ['{}_true'.format(x) for x in label_column_names]

    for forward_pass_idx in tqdm(range(1, config['uncertainty'][UQ_method]['n_forward_passes'] + 1)):
        if UQ_method == "TTA":
            forward_pass_seed = forward_pass_idx # * config['general']['seed']  # set the seed for each forward pass
        else:
            forward_pass_seed = None
        
        # get the predictions for this forward pass
        df_forward_pass_preds = collect_one_predictions_pass(model_config, model, test_loader, enable_MC_dropout=enable_MC_dropout, seed=forward_pass_seed)
        
        df_forward_pass_preds.set_index('PatientID', inplace=True)  # set the index to PatientID
        df_forward_pass_preds.sort_index(inplace=True)  # sort the index by PatientID
        
        # append the predictions to the dataframe
        if forward_pass_idx == 1:
            df_forward_pass_preds = df_forward_pass_preds[[*true_label_columns, *prediction_columns]]
            # change the name of the predictions columns to include the model index
            df_forward_pass_preds.columns = [f"{col}_model_{forward_pass_idx}" if col in prediction_columns else col for col in df_forward_pass_preds.columns]
            # add the model's predictions to the dataframe
            DF_ALL_PREDS = df_forward_pass_preds.copy()
            length_df_all_preds = len(DF_ALL_PREDS)

        else:
            assert length_df_all_preds == len(df_forward_pass_preds), "The number of test patients in each model's predictions should be the same."
            assert DF_ALL_PREDS.index.equals(df_forward_pass_preds.index), "The index of the test predictions should be the same for all models."
            
            # remove the true labels from the dataframe
            df_forward_pass_preds = df_forward_pass_preds.drop(columns=true_label_columns)

            # change the name of the predictions columns to include the model index
            df_forward_pass_preds.columns = [f"{col}_model_{forward_pass_idx}" for col in df_forward_pass_preds.columns if col in prediction_columns]
            
            # merge the model's predictions with the all predictions dataframe
            DF_ALL_PREDS = pd.merge(DF_ALL_PREDS, df_forward_pass_preds, left_index=True, right_index=True, how='outer')

    all_prediction_columns = [col for col in DF_ALL_PREDS.columns if "model_" in col]
    all_prediction_columns.sort()
    # reorder the columns to have the true labels first, then the predictions
    DF_ALL_PREDS = DF_ALL_PREDS[true_label_columns + all_prediction_columns]

    # save the predictions to a csv file
    DF_ALL_PREDS.to_csv(os.path.join(experiment_dir, config['uncertainty']['filenames']['all_predictions_filename'] ), sep=';', index=True)

    """
    Save the mean predictions for each toxicity endpoint.
    This will create a new dataframe with the true labels and the mean predictions for each endpoint.
    """
    # make a file with the true labels and mean predictions for each toxicity
    df_mean_predictions = make_mean_predictions_dataframe(DF_ALL_PREDS, endpoint_list)
    
    df_mean_predictions.to_csv(os.path.join(experiment_dir, config['uncertainty']['filenames']['mean_predictions_filename'] ), sep=';', index=True)

    del model, df_mean_predictions, DF_ALL_PREDS




