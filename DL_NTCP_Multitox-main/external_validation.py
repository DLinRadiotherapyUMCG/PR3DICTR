import pandas as pd
import os
import glob
import numpy as np
import torch
import misc
from glob import glob

from data_preproc.data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from utils.calibration_plots import adaptive_make_calibration_plots
from utils import load_data, losses, metrics
from main import validate
from tqdm import tqdm
from models.utils import get_classification_model


def validate_over_N_folds(cfg, logger):
    """
    Function that loops through an experiment folder, and finds the folders for each cross-validation fold
    """
    # variables
    test_aucs_list_dict  = {endpoint: list() for endpoint in cfg.endpoint_list}
    ensemble_DL_predictions = {endpoint: None for endpoint in cfg.endpoint_list}
    test_losses_list = []
    #DL_test_ECE, DL_test_MCE, LR_test_ECE, LR_test_MCE = None, None, None, None
    #DL_ens_ECE, DL_ens_MCE, LR_ens_ECE, LR_ens_MCE = None, None, None, None
    DL_test_ECE_list_dict = {endpoint: list() for endpoint in cfg.endpoint_list}

    exp_root_dir = cfg.exp_root_dir

    # exp_root_dir = os.path.join("experiments", "test_ResNeXt")
    # cfg.model_name = "ResNeXt"

    # find the folder for each cross-validation fold, by using the folders that have the model name in them
    
    all_folder_dirs = glob(exp_root_dir + "/*/") # find all subfolders in this folder
    all_folder_dirs = [os.path.normpath(x) for x in all_folder_dirs] # normalise the paths (from glob-style to os-style)

    folder_dirs = [x for x in all_folder_dirs if cfg.model_name in os.path.split(x)[-1]]  # check if model name in folder name
    
    # Load the dataset here (once)
    _, _, test_dict = load_data.get_files(config=cfg, logger=logger)
    # Define training, internal validation and test transforms
    train_transforms, val_transforms = load_data.get_transforms(config=cfg, logger=logger)
    # make the test set dataloader
    test_dl = load_data.get_dataloader(cfg, test_dict, val_transforms, batch_size=1, drop_last=False, logger=logger, print_info=False, disable_shuffle=True)
    folder_dirs.sort()
    cfg.cv_folds = len(folder_dirs)

    for idx, folder_dir in enumerate(folder_dirs, start=1):
        
        logger.my_print(f"Fold : {idx}/{cfg.cv_folds}")
        cfg.exp_dir = folder_dir

        folder_name = os.path.split(folder_dir)[-1]
        fold_idx = folder_name.split("_")[2]
        # check here to see if the model weights exist
        model_weights_exist = os.path.isfile(os.path.join(folder_dir, cfg.filename_best_model_pth))
        if model_weights_exist:
            test_loss_value, test_loss_dict, test_auc_value_dict, test_avg_auc_value, \
                test_patient_ids, test_y_pred_list_dict, test_y_true_list_dict = validate_one_fold(cfg, folder_dir, test_dl)
            
            logger.add_epoch_dict(name="External test", auc_dict=test_auc_value_dict, loss=test_loss_value, avg_auc=test_avg_auc_value, epoch_n='N/A')
        else:
            raise FileNotFoundError(f"Model weights not present in folder: {folder_dir}")

        """
        process results
        """
        test_losses_list.append(test_loss_value)

        [test_aucs_list_dict] = \
                misc.append_to_list_dicts(config=cfg, list_dicts = [test_aucs_list_dict], 
                                    value_dicts = [test_auc_value_dict])
        # collect the ensemble predictions
        for endpoint in cfg.endpoint_list:
            if ensemble_DL_predictions[endpoint] is None:
                ensemble_DL_predictions[endpoint] = [x for x in test_y_pred_list_dict[endpoint]]
            else:
                ensemble_DL_predictions[endpoint] = [x+y for x,y in zip(ensemble_DL_predictions[endpoint], test_y_pred_list_dict[endpoint])]

            

            

        # calibration and reliability plots
        # make a calibration plot for this fold's validation set
        ensemble_calibration_plot_dict_list = [
            {
                "name" : "DL Model",
                "preds" : test_y_pred_list_dict,
                "labels" : test_y_true_list_dict,
            } #,{                                                                   # TODO: external validation CITOR
            #     "name" : "CITOR",
            #     "preds" : ensemble_LR_predictions,
            #     "labels" : test_y_lr_list_dict,
            # }
        ]
        adaptive_make_calibration_plots(cfg, ensemble_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='calibration', 
                                        title="Calibration Plot: Ensemble (Test Set)", filedir = os.path.join(cfg.exp_dir, 'calibration_plot_external_validation.png'))
        [DL_test_ECE], [DL_test_MCE] = adaptive_make_calibration_plots(cfg, ensemble_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='reliability',
                                        title="Reliability Plot: Ensemble (Test Set)", filedir = os.path.join(cfg.exp_dir, 'reliability_plot_external_validation.png'))
        

        # make_calibration_plots(cfg, predictions_dict=test_y_pred_list_dict, true_labels_dict=test_y_true_list_dict, 
        #                                     lr_predictions_dict=None, lr_true_labels_dict=None, 
        #                                     mode='calibration', set_name="External Validation Set", 
        #                                     filename=os.path.join(cfg.exp_dir, 'calibration_plot_external_validation.png'))
        # DL_test_ECE, DL_test_MCE, LR_test_ECE, LR_test_MCE = make_calibration_plots(cfg, predictions_dict=test_y_pred_list_dict, true_labels_dict=test_y_true_list_dict, 
        #                                     lr_predictions_dict=None, lr_true_labels_dict=None, 
        #                                     mode='reliability', set_name="External Validation Set", 
        #                                     filename=os.path.join(cfg.exp_dir, 'reliability_plot_external_validation.png'))
        
        [DL_test_ECE_list_dict] = \
                misc.append_to_list_dicts(config=cfg, list_dicts =[DL_test_ECE_list_dict], value_dicts = [DL_test_ECE])

        # save results
        results_dict = {"external_test_avg_auc": test_avg_auc_value, "external_test_auc_per_toxicity": test_auc_value_dict,
                        "external_test_mean_loss": test_loss_value, "external_test_loss_per_toxicity": test_loss_dict, 
                        "external_test_avg_ECE": DL_test_ECE, "external_test_avg_MCE": DL_test_MCE}
        misc.append_to_results_file(cfg, **results_dict)
        
        if int(fold_idx) == int(cfg.cv_folds):
            for endpoint in cfg.endpoint_list:
                ensemble_DL_predictions[endpoint] = [x/cfg.cv_folds for x in ensemble_DL_predictions[endpoint]]
            # calculate metrics over the ensembled predictions
            ens_avg_auc_value, ens_auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, ensemble_DL_predictions, test_y_true_list_dict)

            logger.add_epoch_dict(name="Ensemble external test", auc_dict=ens_auc_value_dict, loss='N/A', avg_auc=ens_avg_auc_value, epoch_n='N/A')

            mode_list = ['test'] * len(test_patient_ids)
            misc.save_predictions(config=cfg, patient_ids=test_patient_ids, endpoint_list=cfg.endpoint_list,
                                    y_pred_list_dict=ensemble_DL_predictions, y_true_list_dict=test_y_true_list_dict,  # test_y_true_list_dict should be the same for all folds!
                                    mode_list=mode_list, model_name=cfg.model_name + '_external_ens',
                                    exp_dir=cfg.exp_dir, logger=logger) 



            mean_test_auc_all_folds,  mean_test_aucs_per_toxicity_all_folds  = misc.summarise_results_over_folds(cfg, test_aucs_list_dict, test_set=True)
            DL_mean_test_ECE,  DL_mean_test_ECE_per_toxicity_all_folds       = misc.summarise_results_over_folds(cfg, DL_test_ECE_list_dict, test_set=True)

            # make_calibration_plots(cfg, predictions_dict=ensemble_DL_predictions, true_labels_dict=test_y_true_list_dict, 
            #                             lr_predictions_dict=None, lr_true_labels_dict=None, 
            #                             mode='calibration', set_name="External Validation Set (Ensemble)", 
            #                             filename=os.path.join(cfg.exp_dir, 'calibration_plot_external_validation_ensemble.png'))
            # DL_ens_ECE, DL_ens_MCE, LR_ens_ECE, LR_ens_MCE = make_calibration_plots(cfg, predictions_dict=ensemble_DL_predictions, true_labels_dict=test_y_true_list_dict, 
            #                         lr_predictions_dict=None, lr_true_labels_dict=None, 
            #                         mode='reliability', set_name="External Validation Set (Ensemble)", 
            #                         filename=os.path.join(cfg.exp_dir, 'reliability_plot_external_validation_ensemble.png'))
            ensemble_calibration_plot_dict_list = [
                {
                    "name" : "DL Model",
                    "preds" : ensemble_DL_predictions,
                    "labels" : test_y_true_list_dict,
                } #,{                                                                   # TODO: external validation CITOR
                #     "name" : "CITOR",
                #     "preds" : ensemble_LR_predictions,
                #     "labels" : test_y_lr_list_dict,
                # }
            ]
            adaptive_make_calibration_plots(cfg, ensemble_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='calibration', 
                                            title="Calibration Plot: External Validation Set (Ensemble)", filedir = os.path.join(cfg.exp_dir, 'calibration_plot_external_validation_ensemble.png'))
            [DL_ens_ECE], [DL_ens_MCE] = adaptive_make_calibration_plots(cfg, ensemble_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='reliability',
                                            title="Reliability Plot: External Validation Set (Ensemble)", filedir = os.path.join(cfg.exp_dir, 'reliability_plot_external_validation_ensemble.png'))
        
            

            # TODO: save the ensemble predictions and results here
            mean_test_results_dict = {"external_test_avg_auc": mean_test_auc_all_folds, "external_test_avg_auc_per_toxicity": mean_test_aucs_per_toxicity_all_folds,
                                "external_test_avg_loss": np.mean(test_losses_list), "external_test_avg_ECE": DL_mean_test_ECE, "external_test_avg_ECE_per_toxicity": DL_mean_test_ECE_per_toxicity_all_folds}

            ens_results_dict = {'external_ens_avg_auc_value' : ens_avg_auc_value, 'external_ens_auc_value_dict' : ens_auc_value_dict,
                                    'DL_ens_ECE' : DL_ens_ECE, 'DL_ens_MCE' : DL_ens_MCE}
            
            mean_test_results_dict.update(ens_results_dict)

            misc.append_to_ensemble_results_file(cfg, **mean_test_results_dict)

        
        logger.print_epoch_table(mode="Final Model")
    




    


def validate_one_fold(cfg, folder_dir, dataloader):
    device = cfg.device
    channels = len(cfg.image_keys)
    [depth, height, width] = cfg.input_size 
    n_features = len(cfg.features_dl)
    
    # load the model
    pretrained_path = os.path.join(folder_dir, cfg.filename_best_model_pth)
    
    model, total_model_params = get_classification_model(config=cfg, channels=3, depth=depth, height=height, width=width, n_features=n_features, 
                           pretrained_path=pretrained_path, logger=logger, save_summary=False)
    model.to(device)
    #model.eval()

    loss_function = losses.get_loss_function(config=cfg, train_dict=None, val_dict=None)
    

    test_loss_value, test_loss_dict, test_auc_value_dict, test_avg_auc_value, test_patient_ids, test_y_pred_list_dict, test_y_true_list_dict = \
                validate(cfg=cfg, model=model, dataloader=dataloader, 
                         mode='test', loss_function=loss_function, logger=logger, save_outputs=True)
    

    # save the results for this fold

    return test_loss_value, test_loss_dict, test_auc_value_dict, test_avg_auc_value, test_patient_ids, test_y_pred_list_dict, test_y_true_list_dict
    

    



if __name__ == '__main__':  
    import config as config_file

    cfg = misc.make_config_object(config_file)


    logger = Logger(output_filename="validation_log", suppress_CLI_prints=False)

    validate_over_N_folds(cfg, logger)