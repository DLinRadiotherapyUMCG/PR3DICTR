import os
import logging
import numpy as np

from src.constants import DEVICE

# from src.training.train import train
from src.training.train_AMP import train

from src.training.validate import validate
from src.dataset.load_dataset import load_dataset, generate_K_fold_cross_validation_splits
from src.dataset.cumulative_sampling import generate_training_data_subsamples
from src.dataset.LabelTypesManager import LabelTypesManager as LabelTypesManagerClass
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.get_transforms import get_transforms
from src.models.tools.get_classification_model import get_classification_model
from src.utils.clear_cache import clear_cache
from src.utils.loss_func.get_loss_function import get_loss_function
from src.utils.saving.saving_predictions import concatenate_predictions, save_predictions
from src.utils.saving.create_results_directory import create_results_directory
from src.utils.list_dicts import append_to_list_dicts
from src.utils.data_equalizer import  label_equalizer
from src.config_presets.tools.save_config import save_config
from src.hyper_opt.WandB_functions import initialise_WandB_group, stop_WandB_trial
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.evaluation.total_evaluation import total_evaluation_current_fold
from src.evaluation.aggregate_metrics import aggregate_cross_validation_metrics
from src.evaluation.get_visualisations import get_visualizations

def K_fold_cross_validation(config, config_for_wandb=None, modelCard = None):
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
    LabelTypesManager = LabelTypesManagerClass(config)
    config['saving']['label_column_names'] = LabelTypesManager.label_names_full_list  # save the label column names in the config for saving predictions
    metric_name = config['evaluation']['main_metric']

    # variables for results logging
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
    loss_function = get_loss_function(config, LabelTypesManager)
    metricHandler = mainMetricHandler(config, LabelTypesManager)  # deals with the main metric to print during training

    # iterate through the folds 
    for fold_idx, dataset_split_dict in enumerate(k_fold_dataframes_list, start=1):
        clear_cache()  # clear the CPU memory

        # set up logging and create the results directory
        create_results_directory(config, fold_idx)
        logging.info(f'Fold {fold_idx}/{len(k_fold_dataframes_list)}')
        initialise_WandB_group(config, project_name=config['general']['experiment_name'], groupName=config['general']['trialNumber'], config_for_wandb=config_for_wandb)
        
        """
        MODEL TRAINING
        """

        # get the data split and make the dataloaders for this fold
        train_data, val_data = dataset_split_dict['train'], dataset_split_dict['val']
        # perform over/undersampling of the training set here
        if config['data']['equalizer']['isEnabled']:
            train_data = label_equalizer(train_data, config)
        train_loader, metadata = make_dataloader(config, train_data, train_transforms, validation_mode=False)
        val_loader, _ = make_dataloader(config, val_data, val_transforms, validation_mode=True)

        # initialise a model
        logging.info('Getting model')
        model = get_classification_model(config, metadata=metadata, save_summary=True)
        model.to(device=DEVICE)
        #model = torch.compile(model)

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
        logging.info(f'   Mean {metric_name}: {train_mean_metric_val}')
        logging.info(f'   Loss: {train_loss_value}')
        logging.info(f'   Metrics: {train_metric_dict}')
        logging.info('   Validation set')
        val_loss_value, val_loss_dict, val_mean_metric_val, val_metric_dict, val_preds_dict, val_targets_dict, val_patientIDs_list = validate(config, model, loss_function, val_loader, metricHandler)
        print("   ",val_loss_value, val_mean_metric_val, val_metric_dict)
        logging.info(f'   Mean {metric_name}: {val_mean_metric_val}')
        logging.info(f'   Loss: {val_loss_value}')
        logging.info(f'   Metrics: {val_metric_dict}')

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
            test_preds_dict = {endpoint: None for endpoint in endpoint_list}
            test_targets_dict = {endpoint: None for endpoint in endpoint_list}
        

        """    
        HANDLING RESULTS 
        """

        # concatenate all of the predictions
        all_patientIDs_list = train_patientIDs_list + val_patientIDs_list + test_patientIDs_list                 # IDs column
        mode_list = ['train']*len(train_patientIDs_list) + ['val']*len(val_patientIDs_list) + ['test']*len(test_patientIDs_list)  # Mode column
        all_preds_dict, all_targets_dict = concatenate_predictions(config, [train_preds_dict, val_preds_dict, test_preds_dict],  # concat the predictions and labels
                                                                   [train_targets_dict, val_targets_dict, test_targets_dict])

        # save all the predictions into one csv file
        save_predictions(config, LabelTypesManager, all_patientIDs_list, all_preds_dict, all_targets_dict, mode_list)


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

            if (mean_val_metric_value < config['hyperparam_tuning']['optuna']['kill_trial_threshold']) and (fold_idx >= 3):
                logging.info(f'Early stopping at fold {fold_idx}. Metric value is too low')
                break

        if config['data']['dataloader']['dataset_type'] == 'smartcache':
            train_loader.dataset.shutdown()  # shutdown the smartcache dataloader
            val_loader.dataset.shutdown() 

        del train_loader, val_loader # delete the dataloaders to free up memory
        clear_cache()  # clear the CPU memory
    
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

    if(modelCard != None):
        pathExport = os.path.join(config['modelcard']['saveLocation'],config['modelcard']['saved_modelcard_filename'] + ".json")
        modelCard.absorb_config_details(config)
        modelCard.to_json(pathExport)

    return results

    

