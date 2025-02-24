import os
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from src.constants import DEVICE
from src.training.train import train
from src.training.validate import validate
from src.dataset.load_dataset import load_dataset, generate_K_fold_cross_validation_splits
from src.models.tools.get_classification_model import get_classification_model
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.get_transforms import get_transforms
from src.utils.loss_func.get_loss_function import get_loss_function
from src.utils.saving.saving_predictions import concatenate_predictions, save_predictions
from src.utils.saving.create_results_directory import create_results_directory
from src.utils.list_dicts import append_to_list_dicts

from src.config_presets.tools.save_config import save_config
from src.hyper_opt.WandB_functions import initialise_WandB_group, login, stop_WandB_trial
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.evaluation.total_evaluation import total_evaluation_current_fold
from src.evaluation.aggregate_metrics import aggregate_cross_validation_metrics
from src.evaluation.get_visualisations import get_visualizations





def perform_cumulative_sampling(config, df, sampled_indices, target_sample_size):
    """
    """
    
    remaining_indices = list(set(range(len(df))) - set(sampled_indices))  # determine which patients have not already been sampled
    n_required_samples = target_sample_size - len(sampled_indices)  # how many more patients  we need to sample
    

    if n_required_samples == len(remaining_indices):
        new_indices = remaining_indices
    elif n_required_samples < 0:
        # if the number of required samples is less than 0, raise an error
        # i.e. there is not enough training data to sample the target number of patients
        raise ValueError(f"Number of required samples is less than 0. There is not enough training data to sample {target_sample_size} patients.")
    else:
        if config["data"]["kFolds"]["split_strategy"] == 'stratified':
            labels = df.iloc[remaining_indices][config["data"]["kFolds"]["split_strategy"] == 'stratified'] 
            encoded_labels = LabelEncoder().fit_transform([''.join(str(l)) for l in labels.values])
            shuffle_split = StratifiedShuffleSplit(n_splits=1, train_size=n_required_samples)
            new_indices, _ = next(shuffle_split.split(df[remaining_indices], encoded_labels[remaining_indices]))
            new_indices = np.array(remaining_indices)[new_indices]

            # strat_split = StratifiedShuffleSplit(n_splits=1, train_size=required_samples, random_state=random_state)
            # new_indices, _ = next(strat_split.split(X[remaining_indices], y[remaining_indices]))
            # new_indices = np.array(remaining_indices)[new_indices]

        else:
            # shuffle the indicies of the remaining patients and select the required number of samples
            shuffle_split = ShuffleSplit(n_splits=1, train_size=n_required_samples) # , random_state=config['general']['seed'])
            new_indices, _ = next(shuffle_split.split(remaining_indices))
            new_indices = np.array(remaining_indices)[new_indices]  # Convert back to the original index space

    # update the list of sampled indices with the new_indices
    sampled_indices += list(new_indices)

    # check that the number of unique indices is equal to the target sample size
    assert len(sampled_indices) == len(set(sampled_indices)) == target_sample_size
    
    return df.iloc[sampled_indices], sampled_indices



def generate_training_data_subsamples(config, k_fold_dataframes_list):
    dataset_split_dict = k_fold_dataframes_list[0]
    df_all_train_patients, df_val = dataset_split_dict['train'], dataset_split_dict['val']

    k_fold_dataframes_list = []
    sampled_indices = []
        
    for n_training_patients in config['data']['n_training_patients_list']:
            #config["data"]['n_training_patients'] = n_training_patients
        df_train_sample, sampled_indices = perform_cumulative_sampling(config, df_all_train_patients, sampled_indices, n_training_patients)

        k_fold_dataframes_list.append({'train': df_train_sample, 'val': df_val})
        print(f"Sampled {len(df_train_sample)} patients")

    print(len(k_fold_dataframes_list), len(config['data']['n_training_patients_list']))
    return k_fold_dataframes_list













def K_fold_cross_validation(config, config_for_wandb=None):
    """
    A function to perform K-folds cross-validation. Creates the dataset splits, trains and evaluates a model for N folds.
    Args:
        config (dict): the config params.
        config_for_wandb (dict): a config to send to WandB (optional, usually only used during hyperparameter tuning).
    Returns:
        results (dict): the mean results over each fold.

    """

    n_splits = config['data']['kFolds']['n_splits']
    n_iterations = config['data']['kFolds']['n_iterations']
    if n_splits < n_iterations:
        raise ValueError("The number of k-fold splits must be greater than or equal to the number of k-fold iterations.")
    elif n_iterations == 0:
        raise ValueError("The number of k-fold iterations must be greater than 0.")

    endpoint_list = config['columns']['labels']
    metric_name = config['evaluation']['main_metric']

    # variables for results logging
    train_aucs_list = []
    val_aucs_list = []
    test_aucs_list = []
    train_metrics_list_dict = {endpoint: list() for endpoint in endpoint_list}
    val_metrics_list_dict   = {endpoint: list() for endpoint in endpoint_list}
    test_metrics_list_dict  = {endpoint: list() for endpoint in endpoint_list}

    train_losses_list = []
    val_losses_list = []
    test_losses_list = []
    train_losses_list_dict = {endpoint: list() for endpoint in endpoint_list}
    val_losses_list_dict   = {endpoint: list() for endpoint in endpoint_list}
    test_losses_list_dict  = {endpoint: list() for endpoint in endpoint_list}
    


    # load the data, and make K-fold splits
    df_train_val, df_test = load_dataset(config)
    k_fold_dataframes_list = generate_K_fold_cross_validation_splits(config, df_train_val)
    
    # cap the number of iterations, if it is less than the number of k-splits to make
    n_iterations = config['data']['kFolds']['n_iterations']
    if n_iterations < len(k_fold_dataframes_list):
        k_fold_dataframes_list = k_fold_dataframes_list[:n_iterations]

    # if we're doing the dataset amounts experiment, then make each fold be a different number of training patients (using just the first iteration of the K-fold split)
    if config['general']['dataset_amounts_experiment']:
        k_fold_dataframes_list = generate_training_data_subsamples(config, k_fold_dataframes_list)

    # get the data transforms
    train_transforms, val_transforms = get_transforms(config)
    if config['general']['use_test_set']:
        test_loader, _ = make_dataloader(config, df_test, val_transforms, validation_mode=True)
        
    # get the loss function and metric handler
    loss_function = get_loss_function(config)
    metricHandler = mainMetricHandler(config)  # deals with the main metric to print during training

    # iterate through the folds 
    for fold_idx, dataset_split_dict in enumerate(k_fold_dataframes_list, start=1):

        # set up logging and create the results directory
        create_results_directory(config, fold_idx)
        logging.info(f'Fold {fold_idx}/{len(k_fold_dataframes_list)}')
        initialise_WandB_group(config, project_name=config['general']['experiment_name'], groupName=config['general']['trialNumber'], config_for_wandb=config_for_wandb)
        

        """
        MODEL TRAINING
        """

        # get the data split and make the dataloaders for this fold
        train_data, val_data = dataset_split_dict['train'], dataset_split_dict['val']
        train_loader, metadata = make_dataloader(config, train_data, train_transforms, validation_mode=False)
        val_loader, _ = make_dataloader(config, val_data, val_transforms, validation_mode=True)

        # if using pos_weight in BCE, then calculate the pos_weight for each class
        """
        if config['training']['loss']['name'] == 'BCE' and config['training']['loss']['BCE']['pos_weight'] == 'auto':
            pos_weight = torch.tensor([(len(train_data) - train_data[endpoint].sum()) / len(train_data) for endpoint in endpoint_list])
            config['loss']['BCE']['pos_weight'] = pos_weight
        """
            #pos_weight = pos_weight.to(DEVICE)
            #loss_function = get_loss_function(config, pos_weight=pos_weight)

        # initialise a model
        logging.info('Getting model')
        model = get_classification_model(config, metadata=metadata, save_summary=True)
        model.to(device=DEVICE)

        # train the model
        model = train(config, model, loss_function, train_loader, val_loader, metricHandler)
        model.eval()

        # get the predictions of the trained model on the training and validation (and test) sets
        logging.info('Getting predictions of best model from this fold:')
        logging.info('   Training set')
        # if MixUp is enabled, then it has to be forcibly disabled before the training set is evaluated
        if config['data']['augmentation']['mixup']['isEnabled']: 
            from monai.data.utils import list_data_collate
            train_loader.dataset.collate_fn = list_data_collate  # replace the mixup function with a simple collate function
        
        train_loss_value, train_loss_dict, train_mean_metric_val, train_metric_dict, train_preds_dict, train_targets_dict, train_patientIDs_list = validate(config, model, loss_function, train_loader, metricHandler)
        print("   ", train_loss_value, train_mean_metric_val, train_metric_dict)
        logging.info('   Validation set')
        val_loss_value, val_loss_dict, val_mean_metric_val, val_metric_dict, val_preds_dict, val_targets_dict, val_patientIDs_list = validate(config, model, loss_function, val_loader, metricHandler)
        print("   ",val_loss_value, val_mean_metric_val, val_metric_dict)

        # if the test set is enabled, also collect the results on that set
        if config['general']['use_test_set']:
            logging.info('   Test set')
            test_loss_value, test_loss_dict, test_mean_metric_val, test_metric_dict, test_preds_dict, test_targets_dict, test_patientIDs_list = validate(config, model, loss_function, test_loader, metricHandler)
            print("   ",test_loss_value, test_mean_metric_val, test_metric_dict)
        else:
            test_loss_value = None
            test_loss_dict = {endpoint: None for endpoint in endpoint_list}
            test_mean_metric_val = None
            test_metric_dict = {endpoint: None for endpoint in endpoint_list}
            test_patientIDs_list = []
            test_preds_dict = {endpoint: [] for endpoint in endpoint_list}
            test_targets_dict = {endpoint: [] for endpoint in endpoint_list}
        

        """    
        HANDLING RESULTS 
        """

        # concatenate all of the predictions
        all_patientIDs_list = train_patientIDs_list + val_patientIDs_list + test_patientIDs_list                 # IDs column
        mode_list = ['train']*len(train_patientIDs_list) + ['val']*len(val_patientIDs_list) + ['test']*len(test_patientIDs_list)  # Mode column
        all_preds_dict, all_targets_dict = concatenate_predictions(config, [train_preds_dict, val_preds_dict, test_preds_dict],  # concat the predictions and labels
                                                                   [train_targets_dict, val_targets_dict, test_targets_dict])

        # save all the predictions into one csv file
        save_predictions(config, all_patientIDs_list, all_preds_dict, all_targets_dict, mode_list)


        # concatenate all of the AUC dicts and loss dicts
        [train_metrics_list_dict, val_metrics_list_dict, test_metrics_list_dict] = append_to_list_dicts(config, 
                                                                                               [train_metrics_list_dict, val_metrics_list_dict, test_metrics_list_dict], 
                                                                                               [train_metric_dict, val_metric_dict, test_metric_dict]
                                                                                               )

        [train_losses_list_dict, val_losses_list_dict, test_losses_list_dict] = append_to_list_dicts(config,
                                                                                                         [train_losses_list_dict, val_losses_list_dict, test_losses_list_dict],
                                                                                                         [train_loss_dict, val_loss_dict, test_loss_dict]
                                                                                                         )
        
        train_losses_list.append(train_loss_value)
        val_losses_list.append(val_loss_value)
        test_losses_list.append(test_loss_value)

        # stop WandB for this fold (init a new one on the next fold)
        stop_WandB_trial(config)

        # save the config file for this fold
        save_config(config)

        sets_to_evaluate = ['train', 'val']
        if config['general']['use_test_set']:
            sets_to_evaluate.append('test')

        # collect all metrics for this fold
        total_evaluation_current_fold(config, sets=sets_to_evaluate, external_set=False)

        # make the visualisations (plots) for this fold
        get_visualizations(config, sets=sets_to_evaluate, pred_csv_dir=None, external_set=False)

        # to kill optuna trials early, check the mean metrics for this fold
        if config['hyperparam_tuning']['optuna']['isEnabled']:
            val_metrics_mean_dict = {endpoint: np.mean(aucs) for endpoint, aucs in val_metrics_list_dict.items()}
            mean_val_metric_value = np.mean(list(val_metrics_mean_dict.values()))

            if (val_mean_metric_val < config['hyperparam_tuning']['optuna']['kill_trial_threshold']) and (fold_idx >= 3):
                logging.info(f'Early stopping at fold {fold_idx}. Metric value is too low')
                break
    
    # if we're doing the dataset amounts experiment, then we don't need to aggregate the results. Just return here
    if config['general']['dataset_amounts_experiment'] == True:
        return 
    

    # compute mean AUC per endpoint
    train_metrics_mean_dict = {endpoint: np.mean(aucs) for endpoint, aucs in train_metrics_list_dict.items()}
    val_metrics_mean_dict = {endpoint: np.mean(aucs) for endpoint, aucs in val_metrics_list_dict.items()}
    # compute total mean AUC
    mean_train_metric_value = np.mean(list(train_metrics_mean_dict.values()))
    mean_val_metric_value = np.mean(list(val_metrics_mean_dict.values()))

    # compute mean loss per endpoint
    train_mean_losses_dict = {endpoint: np.mean(losses) for endpoint, losses in train_losses_list_dict.items()}
    val_mean_losses_dict = {endpoint: np.mean(losses) for endpoint, losses in val_losses_list_dict.items()}
    # compute total mean loss
    mean_train_loss = np.mean(train_losses_list)
    mean_val_loss = np.mean(val_losses_list)

    # do that for the test set
    if config['general']['use_test_set']:
        mean_test_metric_dict = {endpoint: np.mean(aucs) for endpoint, aucs in test_metrics_list_dict.items()}
        mean_test_metric_value = np.mean(list(mean_test_metric_dict.values()))
        test_mean_losses_dict = {endpoint: np.mean(losses) for endpoint, losses in test_losses_list_dict.items()}
        mean_test_loss = np.mean(test_losses_list)
    else:
        mean_test_metric_dict = {endpoint: None for endpoint in endpoint_list}
        mean_test_metric_value = None
        test_mean_losses_dict = {endpoint: None for endpoint in endpoint_list}
        mean_test_loss = None
    
    results = { 
                # mean AUC
                f"train_mean_{metric_name}": mean_train_metric_value,
                f"val_mean_{metric_name}": mean_val_metric_value,
                f"test_mean_{metric_name}": mean_test_metric_value,
                # mean Loss
                "train_mean_loss": mean_train_loss,
                "val_mean_loss": mean_val_loss,
                "test_mean_loss": mean_test_loss,
              }
    
    for endpoint in endpoint_list:
        # mean AUC per endpoint
        results[f"train_{endpoint}_{metric_name}"] = train_metrics_mean_dict[endpoint]
        results[f"val_{endpoint}_{metric_name}"] = val_metrics_mean_dict[endpoint]
        results[f"test_{endpoint}_{metric_name}"] = mean_test_metric_dict[endpoint]

        # mean loss per endpoint
        results[f"train_{endpoint}_loss"] = train_mean_losses_dict[endpoint]
        results[f"val_{endpoint}_loss"] = val_mean_losses_dict[endpoint]
        results[f"test_{endpoint}_loss"] = test_mean_losses_dict[endpoint]

    # aggregate all of the metric results for this trial
    aggregate_cross_validation_metrics(config, k_folds_completed=fold_idx, sets=['train', 'val'])

    return results

    

