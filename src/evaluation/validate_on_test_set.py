import os    
import torch
import copy
import pandas as pd

from src.utils.saving.saving_predictions import save_predictions
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.utils.loss_func.get_loss_function import get_loss_function
from src.dataset.load_dataset import load_dataset
from src.dataset.get_transforms import get_transforms
from src.dataset.get_dataloader import make_dataloader
from src.training.validate import validate
from src.models.tools.save_model import load_model
from src.evaluation.get_evaluation_metric import get_metric_function
from src.constants import METRIC_TYPES
from src.evaluation.per_endpoint_metrics import calculate_metric_for_multiple_endpoints
from src.evaluation.utils.get_predictions_and_labels_from_predictions_dataframe import get_predictions_and_labels_from_predictions_dataframe    

from src.config_presets.tools.load_config import load_config
from src.models.tools.get_classification_model import get_classification_model
from src.evaluation.get_visualisations import get_visualizations
from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir
from src.evaluation.total_evaluation import total_evaluation_current_fold


def validate_models_on_test_set(main_config, trial_dir = r'\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Luuk vd Hoek\Projects collection\Mini projects\SIMS\hnc-ensemble-master\results\Resnet_optimization_modtype3_2a_depth_newlabels\Trial_19/'): 
    test_dataset_source = main_config['data']['source']
    print(f"Validating models on {test_dataset_source} test set")
    # Get paths to folds and config from the experiment
    folds = [f.path for f in os.scandir(trial_dir) if f.is_dir()] # This does asume all folders within a trial are folds
    config_path = os.path.join(folds[0], main_config['saving']['filenames']['config_yaml']) 
    print(config_path)
    model_config = load_config(config_path) # Gets the config folder from the first fold
    
    # Load data and helper functions
    # the dataloader settings are taken from the model config
    metricHandler = mainMetricHandler(model_config)
    loss_function = get_loss_function(model_config)
    train_transforms, val_transforms = get_transforms(model_config)

    # the dataset itself is loaded from the main config (this allows for using the external test set)
    df_train_val, df_test = load_dataset(main_config)
    test_loader, metadata = make_dataloader(main_config, df_test, val_transforms, validation_mode=True)


    # used later, stores location of generated predictions
    prediction_paths = []
    test_combined_preds = None
    test_patientIDs_list_fold1 = None
    
    """
    TEST SET RESULTS PER FOLD
    """
    # Generate and save predictions for each fold

    print(folds)
    for fold_dir in folds:

        config_path = os.path.join(fold_dir, main_config['saving']['filenames']['config_yaml']) 
        model_config = load_config(config_path)

        model = get_classification_model(model_config, metadata, save_summary=False)
        model.cuda()
        model = load_model(model_config, model) # load the saved weights
    
        # get the model predictions on this set
        test_loss, test_loss_dict, test_mean_metric_val, test_metric_dict, test_preds_dict, test_targets_dict, test_patientIDs_list = validate(model_config, model, loss_function, test_loader, metricHandler)
        
        # check to see if the patient IDs are the same order for all folds
        if test_patientIDs_list_fold1 == None:
            test_patientIDs_list_fold1 = test_patientIDs_list
        else:
            assert test_patientIDs_list_fold1 == test_patientIDs_list, 'Patient IDs are not the same for all folds'

        # save the predictions of this fold
        mode_list = ['test'] * len(test_patientIDs_list)
        main_config['general']['resultsCurrentDirectory'] = fold_dir + r'/' # fold results saves to fold
        prediction_paths.append(main_config['general']['resultsCurrentDirectory'])
        save_predictions(main_config, test_patientIDs_list, test_preds_dict, test_targets_dict, mode_list, is_test_set=True) 
        
        # save the predictions, to use for the ensemble predictions later
        if test_combined_preds == None:
            test_combined_preds = test_preds_dict
        else:
            for key in test_combined_preds.keys():
                test_combined_preds[key] = [x + y for x, y in zip(test_combined_preds[key], test_preds_dict[key])]


        # compute metrics and save for this fold's test set
        pred_csv_dir = get_predictions_csv_dir(main_config, test_set=True, ensemble_predictions=False)
        total_evaluation_current_fold(main_config, sets = ['test'], pred_csv_dir = pred_csv_dir)

        # now create the visualisations
        get_visualizations(main_config, sets=['test'], pred_csv_dir=pred_csv_dir)
    
    """
    ENSEMBLE RESULTS
    """

    # Combine the test set results 
    for key in test_combined_preds.keys():
        # Divide the combined predictions by the number of folds
        test_combined_preds[key] = [x / len(folds) for x in test_combined_preds[key]]
    
    # Save the ensemble predictions
    mode_list = ['test'] * len(test_patientIDs_list)
    main_config['general']['resultsCurrentDirectory'] = trial_dir # combined results saves to the trial 
    prediction_paths.append(main_config['general']['resultsCurrentDirectory'])
    save_predictions(main_config, test_patientIDs_list, test_combined_preds, test_targets_dict, mode_list, is_test_set=True, ensemble_predictions=True) 
    

    # compute metrics and save for the ensmble predictions
    pred_csv_dir = get_predictions_csv_dir(main_config, test_set=True, ensemble_predictions=True)
    total_evaluation_current_fold(main_config, sets = ['test'], pred_csv_dir = pred_csv_dir)

    # now create the visualisations
    get_visualizations(main_config, sets=['test'], pred_csv_dir=pred_csv_dir)
    
    