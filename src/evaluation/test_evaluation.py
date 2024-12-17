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

from src.evaluation.get_evaluation_metric import get_metric_function
from src.constants import METRIC_TYPES
from src.evaluation.per_endpoint_metrics import calculate_metric_for_multiple_endpoints
from src.evaluation.utils.get_predictions_and_labels_from_predictions_dataframe import get_predictions_and_labels_from_predictions_dataframe    

from src.config_presets.tools.load_config import load_config
from src.models.tools.get_classification_model import get_classification_model
from src.evaluation.get_visualisations import get_visualizations


def test_evaluation(trial_path = r'\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Luuk vd Hoek\Projects collection\Mini projects\SIMS\hnc-ensemble-master\results\Resnet_optimization_modtype3_2a_depth_newlabels\Trial_19/'): 
    
    # Get paths to folds and config from the experiment
    folds = [f.path for f in os.scandir(trial_path) if f.is_dir()] # This does asume all folders within a trial are folds 
    config = load_config(folds[0] + '/DlModel_Config.yaml') # Gets the config folder from the first fold
    
    # Load data and helper functions
    metricHandler = mainMetricHandler(config)
    loss_function = get_loss_function(config)
    df_train_val, df_test = load_dataset(config)
    train_transforms, val_transforms = get_transforms(config)
    test_loader, metadata = make_dataloader(config, df_test, val_transforms, validation_mode=True)
    
    # used later, stores location of generated predictions
    prediction_paths = []
    test_combined_preds = None
    test_patientIDs_list_fold1 = None
    
    # Generate and save predictions for each fold
    for path_fold in folds:
        model = get_classification_model(config, metadata, save_summary=False)
        model.cuda()
        print('loading model ...')
        print(path_fold + '/DlModel_Weights.pth')
        model.load_state_dict(torch.load(path_fold + '/DlModel_Weights.pth'))
        print('model loaded!')
    
        test_loss, test_mean_metric_val, test_metric_dict, test_preds_dict, test_targets_dict, test_patientIDs_list = validate(config, model, loss_function, test_loader, metricHandler)
        
        # check to see if the patient IDs are the same order for all folds
        if test_patientIDs_list_fold1 == None:
            test_patientIDs_list_fold1 = test_patientIDs_list
        else:
            assert test_patientIDs_list_fold1 == test_patientIDs_list, 'Patient IDs are not the same for all folds'

        mode_list = ['test'] * len(test_patientIDs_list)
        config['general']['resultsCurrentDirectory'] = path_fold + r'/' # fold results saves to fold
        prediction_paths.append(config['general']['resultsCurrentDirectory'])
        save_predictions(config, test_patientIDs_list, test_preds_dict, test_targets_dict, mode_list, test_set=True) 
        
        if test_combined_preds == None:
            test_combined_preds = test_preds_dict
        else:
            for key in test_combined_preds.keys():
                test_combined_preds[key] = [x + y for x, y in zip(test_combined_preds[key], test_preds_dict[key])]
               
    # Combine the test set results 
    for key in test_combined_preds.keys():
        # Divide the combined predictions by the number of folds
        test_combined_preds[key] = [x / len(folds) for x in test_combined_preds[key]]
        

    # Save the total predictions
    mode_list = ['test'] * len(test_patientIDs_list)
    config['general']['resultsCurrentDirectory'] = trial_path # combined results saves to the trial 
    prediction_paths.append(config['general']['resultsCurrentDirectory'])
    save_predictions(config, test_patientIDs_list, test_combined_preds, test_targets_dict, mode_list, test_set=True) 
    
    # Go through the prediction csvs to generate the metric 
    for pred_path in prediction_paths:
        # Set paths and constants
        config['general']['resultsCurrentDirectory'] = pred_path
        pred_path = pred_path +  r"/model_predictions_test.csv"
        set_name = 'test'
        metrics_to_calculate = config['evaluation']['metrics_list']

        print("metrics_to_calculate", pred_path)
        
        # read the current csv
        test_all_preds = pd.read_csv(pred_path, sep=";")

        # Read the predictions 
        predictions_per_endpoint_dict, labels_per_endpoint_dict = get_predictions_and_labels_from_predictions_dataframe(config, test_all_preds, set_name)
        
        # loop over the metric types
        combined_metrics_df = None
        for metric_type, metric_names in METRIC_TYPES.items():
            for metric_name in list(metric_names):

                if metric_name in metrics_to_calculate:
                    #print("     HELLO   ")
                            
                    # get the metric function
                    metric_function = get_metric_function(metric_name)
        
                    # calculate the metric for each endpoint
                    mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(config, predictions_per_endpoint_dict, labels_per_endpoint_dict, metric_function)
        
                    # store the results in a dataframe
                    results_df = pd.DataFrame(results_dict, index=[metric_name])
                    results_df.insert(0, "MEAN", mean_metric_value)  # put the mean value in the first column of the dataframe
        
                    # append the results to a combined dataframe
                    if 'combined_metrics_df' not in locals(): # if it doesn't exist yet
                        combined_metrics_df = copy.deepcopy(results_df)
                    else:
                        combined_metrics_df = pd.concat([combined_metrics_df, copy.deepcopy(results_df)])
        
        # Set the saving directory
        # # save the combined results to a csv
        combined_metrics_df = combined_metrics_df.sort_index()
        combined_metrics_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], f"all_{set_name}_metrics.csv")
        combined_metrics_df.to_csv(combined_metrics_csv_dir, sep=";")


        # now create the visualisations
        get_visualizations(config, sets=['test'], pred_csv_dir=pred_path, test_set=True)