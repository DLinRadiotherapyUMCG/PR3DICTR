import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'data_preproc'))
import matplotlib
matplotlib.use('Agg')

import time
import math
import torch
import shutil
import numpy as np
from datetime import datetime
import pathlib
import gc

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

import hyperparam_config as hp_cfg
os.environ["RAY_DEDUP_LOGS"] = '0'
os.environ["RAY_AIR_NEW_OUTPUT"] = '0' # must be zero in order for the custom loggers to be used
os.environ['TUNE_RESULT_DIR'] = pathlib.PureWindowsPath(hp_cfg.storage_path).as_posix()  # enforce that Ray doesn't save another copy of the results to ~/ray_results ('home' folder on habrok)
os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'  # enforce that Ray changes the working directory to the trial directory

import optuna
import ray
from ray import train, tune
from ray.train.torch import get_device
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
#import tempfile
#from ray.train import Checkpoint, CheckpointConfig

#from torch.backends.cudnn import benchmark as cudnn_benchmark # RayTune workaround
# Load local functions
import misc
import utils.losses.get_loss as losses
import utils.metrics as metrics
import utils.load_data as load_data
import data_preproc.data_preproc_config as data_preproc_cfg
from data_preproc.data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from models.logistic_regression import refit_logistic_regression
from utils.ray_functions import BasicTrialLogger, TrialTerminationReporter, trial_str_creator, stop_invald_trial, report_invalid_model_trial, rename_folders_for_parameter_tuning
from utils.early_stopping import  load_model_output_head_weights, layer_freezing, freeze_normalisation_layers, early_stopping
from utils.calibration_plots import adaptive_make_calibration_plots
from utils.img_transforms import mixup_data
from models.utils import get_classification_model
from utils.optimizers.get_optimizer import get_optimizer
from utils.get_scheduler import get_scheduler
from utils.config_object import make_config_object





def initialize(config, device, create_folders=True):
    """
    Set up experiment and save the experiment path to be able to save results.
    Args:
        config: the config object
        device: the device to run the model on (also for printing)
        create_folders: whether or not to create the folders for saving the results
    Returns:
        logger: the logger object

    """
    if create_folders:
        # Create experiment folder and subfolders if they do not exist yet
        exp_dir = config.exp_dir
        exp_figures_dir = os.path.join(exp_dir, 'figures')
        create_folder_if_not_exists(exp_dir)
        create_folder_if_not_exists(exp_figures_dir)

        # Logger output filename
        output_filename = os.path.join(exp_dir, 'log.txt')
    
        # do not copy the source files to the HP tuning experiment folders 
        # (it is too many files, and the directories become too long)
        if not config.hyperparameter_tuning_mode:
            exp_src_dir = os.path.join(exp_dir, 'src')
            exp_models_dir = os.path.join(exp_src_dir, 'models')
            exp_optimizers_dir = os.path.join(exp_src_dir, 'optimizers')
            exp_data_preproc_dir = os.path.join(exp_src_dir, 'data_preproc')
            for p in [exp_src_dir, exp_models_dir, exp_optimizers_dir, exp_data_preproc_dir]:
                create_folder_if_not_exists(p)

            # Fetch folders and files that should be copied to exp folder
            root_path = config.root_path
            models_dir = config.models_dir
            optimizers_dir = config.optimizers_dir
            data_preproc_dir = config.data_preproc_dir
            src_files = [x for x in os.listdir(root_path) if
                        (x.endswith('.py') or x == data_preproc_cfg.filename_exclude_patients_csv)]

            # Copy src files to exp folder
            for f in src_files:
                copy_file(src=os.path.join(root_path, f), dst=os.path.join(exp_src_dir, f))
            # Copy folders to exp folder
            copy_folder(src=models_dir, dst=exp_models_dir)
            #copy_folder(src=optimizers_dir, dst=exp_optimizers_dir)     
            copy_folder(src=data_preproc_dir, dst=exp_data_preproc_dir)
    else:
        output_filename = None

    # Initialize logger
    logger = Logger(output_filename=output_filename, suppress_CLI_prints=config.hyperparameter_tuning_mode)

    # Print variables
    logger.my_print(f'Device: {device}.')
    logger.my_print(f'Torch version: {config.torch_version}.')
    logger.my_print(f'Torch.backends.cudnn.benchmark: {config.cudnn_benchmark}.')

    return logger



                


def train_DL_model(cfg, model, train_dataloader, val_dataloader, loss_function, fold_n, logger):
    """
    Trains a model on the training set and validates it on the validation set.
    Args:
        cfg: the config object
        model: the DL model
        train_dataloader: the dataloader of the training set
        val_dataloader: the dataloader of the validation set
        fold_n: the fold number (for printing)
        logger: the logger object
    Returns:
        train_loss_values_list: a list of the loss values for each epoch
        train_avg_auc_values_list: a list of the average AUC values for each epoch
        train_auc_values_list_dict: a dictionary of lists containing the AUC values for each endpoint over all epochs
        val_loss_values_list: a list of the loss values for each epoch
        val_avg_auc_values_list: a list of the average AUC values for each epoch
        val_auc_values_list_dict: a dictionary of lists containing the AUC values for each endpoint over all epochs
        best_val_loss_value: the best validation loss value
        best_val_avg_auc_value: the best validation average AUC value
        best_val_auc_values_dict: a dictionary with the best validation AUC value for each endpoint
        best_val_epoch_num: the epoch number with the best validation results
        lr_list: a list of the learning rates for each epoch
        batch_size_list: a list of the batch sizes for each epoch
        epoch: the last epoch number (i.e. the number of epochs that were actually run)
    """
    # trains the model for X epcohs
    # on each epoch (unless different validation rate), calls test() on the valiation set
    device = cfg.device
    max_epochs = cfg.max_epochs
    endpoint_list = cfg.endpoint_list
    train_num_iterations = len(train_dataloader)

    # Initiate variables
    best_metrics_tracker = metrics.BestMetricsTracker(config=cfg)
    model_backbone_frozen = False                                          # whether or not the model's backbone is frozen
    # the following lists are used to store the results of all epochs (i.e. overview of the whole training process)
    train_avg_auc_values_list = list() if max_epochs > 0 else [None]    
    val_avg_auc_values_list =   list() if max_epochs > 0 else [None]
    train_loss_values_list = list() if max_epochs > 0 else [None]          
    val_loss_values_list =   list() if max_epochs > 0 else [None]
    train_loss_list_dict = {endpoint: list() for endpoint in endpoint_list}
    val_loss_list_dict = {endpoint: list() for endpoint in endpoint_list}
    train_auc_values_list_dict = {endpoint: list() if max_epochs > 0 else [None] for endpoint in endpoint_list}
    val_auc_values_list_dict =   {endpoint: list() if max_epochs > 0 else [None] for endpoint in endpoint_list}
    lr_list = list() if max_epochs > 0 else [None]
    batch_size_list = list() if max_epochs > 0 else [None]
    GPU_RAM_USAGE = list() if max_epochs > 0 else [None]
    mixup_indexes_batch, mixup_lambda = None, None

    # a tensor used to mask the model outputs and labels, so that only the non-missing values are considered
    valid_endpoints_as_tensor = torch.as_tensor(cfg.valid_endpoint_values, device=device)

    #loss_function = losses.get_loss_function(config=cfg, train_dict=train_dict, val_dict=val_dict)
    optimizer = get_optimizer(config=cfg, model=model, num_batches_per_epoch=len(train_dataloader), logger=logger)
    scheduler = get_scheduler(config=cfg, scheduler_name=cfg.scheduler_name, optimizer=optimizer, optimal_lr=cfg.lr,
                                   num_batches_per_epoch=len(train_dataloader), logger=logger)

    logger.my_print(f'Starting fold {fold_n}.')
    logger.my_print(f'Seed: {cfg.seed}.')
    logger.my_print(f'Number of (training) batches per epoch: {train_num_iterations}.')
    for epoch in range(1, max_epochs + 1):
        logger.my_print('-' * 40)
        logger.my_print(f'Epoch {epoch}/{max_epochs}...')
        for param_group in optimizer.param_groups:
            logger.my_print(f"Learning rate: {param_group['lr']}.")
            lr_list.append(param_group['lr'])
        cur_batch_size = train_dataloader.batch_size
        #logger.my_print(f'Batch size: {cur_batch_size}.')
        batch_size_list.append(cur_batch_size)
        if device == torch.device('cuda'):
            mem_free, mem_toal = torch.cuda.mem_get_info()
            GPU_RAM_USAGE.append((mem_toal-mem_free)/1e9)

        # Initiate training variables
        model.train()
        if model_backbone_frozen == True:
            freeze_normalisation_layers(model)
        train_loss_value_epoch = 0
        train_loss_dict = {endpoint: 0 for endpoint in endpoint_list}
        train_y_pred_dict = dict()
        train_y_true_dict = dict()
        mixup_label_indexes = torch.as_tensor([], dtype=torch.int32, device=device)
        mixup_lambda_values = torch.as_tensor([], dtype=torch.float32, device=device)
        for endpoint in endpoint_list:
            train_y_pred_dict[endpoint] = torch.as_tensor([], dtype=torch.float32, device=device)
            train_y_true_dict[endpoint] = torch.as_tensor([], dtype=torch.int8, device=device)
        
        epoch_start_time = time.time()

        for batch_num, batch_data in tqdm(enumerate(train_dataloader), disable=cfg.hyperparameter_tuning_mode):
            ###  1. load the data   ###
                
            # send to device (the data is augmented within the train dataloader)
            train_inputs, train_features, train_label_list = (
                batch_data['ct_dose_seg'].to(device),
                batch_data['features'].to(device),
                batch_data['label_list'].to(device)
            )

            if cfg.perform_mixup:
                # perform mixup augmentation on the images, clinical features, and labels
                cur_batch_size = train_inputs.size(0)
                mixed_images, mixed_features, mixup_lambda, mixup_indexes_batch = mixup_data(cfg, train_inputs, train_features, train_label_list, cur_batch_size, alpha=cfg.mixup_alpha)
                train_inputs = mixed_images
                train_features = mixed_features

                mix_indexes_adjusted = batch_num * cur_batch_size + mixup_indexes_batch
                mixup_lamdas = [mixup_lambda] * cur_batch_size

                mixup_label_indexes = torch.cat([mixup_label_indexes, mix_indexes_adjusted], dim=0)
                mixup_lambda_values = torch.cat([mixup_lambda_values, torch.tensor(mixup_lamdas, dtype=torch.float32, device=device)], dim=0)

                train_label_list[train_label_list == -1] = 0 # mask out the missing labels for the AUC calculation on the training et

            train_label_dict = dict()
            # Note: swap train_label_list (tensor) from shape (batch_size, len(endpoint_list)) to
            # (len(endpoint_list), batch_size), so that each for-loop iteration contains the endpoint's values
            for j, (endpoint, train_label) in enumerate(zip(endpoint_list, torch.swapaxes(train_label_list, 0, 1))):
                train_label_dict[endpoint] = train_label      

            # 1.2 plot the model inputs
            if ((epoch == 1) or (epoch % cfg.plot_interval == 0)) and batch_num==0:
                misc.plot_model_inputs(config=cfg, plot_inputs=train_inputs, epoch=epoch)

            ###  2. train the model  ###
            optimizer.zero_grad(set_to_none = False) # first zero the optimizer gradients 

            # 2.1 feed inputs to the model
            try:
                train_outputs_dict = model(x=train_inputs, features=train_features)  # inputs to model
            except Exception as e:
                logger.my_print("Out of memory error. Skipping trial.")
                if cfg.hyperparameter_tuning_mode:
                    report_invalid_model_trial(config=cfg)
                else:
                    raise e
            
            # 2.2 calculate the loss
            # for SPLC and MLLSC
            if cfg.loss_function_name.lower() in ["splc", "mllsc"]: 
                if epoch >= cfg.change_epoch + 1:
                    if cfg.loss_function_name.lower() == "splc":
                        loss_function.apply_SPLC = True
                    elif cfg.loss_function_name.lower() == "mllsc":
                        loss_function.apply_MLLSC = True

            batch_loss, batch_per_toxicity_loss_dict, \
            unweighted_batch_loss, unweighted_batch_per_toxicity_loss_dict =  losses.calculate_loss(cfg, train_outputs_dict, 
                                                                                                    train_label_dict, valid_endpoints_as_tensor, 
                                                                                                    loss_function, mixup_indexes_batch, mixup_lambda)
            
            # 2.3 backpropagate the loss
            if batch_loss.grad_fn:  # if the loss has a gradient (i.e. it is not only values for missing endpoints)
              if cfg.optimizer_name in ['ada_hessian']:  # this optimiser requires a graph to be created
                  batch_loss.backward(create_graph=True)
              else:
                  batch_loss.backward()

            # Perform gradient clipping
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            if cfg.grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.grad_max_norm)

            # update optimizer
            optimizer.step()
            
            # Reset the .grad fields of our parameters to None after use to break the cycle and avoid the memory leak
            if cfg.optimizer_name in ['ada_hessian']:
                for p in model.parameters():
                    p.grad = None
            # update scheduler
            if cfg.scheduler_name in ['cosine', 'exponential']:  # , 'step']:
                scheduler.step(epoch + (batch_num / train_num_iterations))
            elif cfg.scheduler_name in ['cyclic']:
                scheduler.step()

            # 2.4 save batch results to entire epoch results
            train_loss_value_epoch += unweighted_batch_loss.item() / train_num_iterations   # overall loss
            train_loss_dict = misc.sum_dictionaries(train_loss_dict, unweighted_batch_per_toxicity_loss_dict, divisor=train_num_iterations)

            for endpoint in endpoint_list:
                train_y_pred_dict[endpoint] = torch.cat([train_y_pred_dict[endpoint], train_outputs_dict[endpoint]], dim=0)
                train_y_true_dict[endpoint] = torch.cat([train_y_true_dict[endpoint], train_label_dict[endpoint]], dim=0)

            del batch_loss, batch_per_toxicity_loss_dict, unweighted_batch_loss, unweighted_batch_per_toxicity_loss_dict, \
                train_inputs, train_features, train_label_list, batch_data, train_outputs_dict, train_label_dict

        epoch_end_time = time.time()
        logger.my_print(f'Epoch {epoch} duration: {epoch_end_time - epoch_start_time:.2f} seconds.')

        # Compute training AUC
        y_pred_list_dict, y_true_list_dict = misc.apply_sigmoid_or_softmax(cfg, train_y_pred_dict, train_y_true_dict)

        train_avg_auc_value, train_auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, y_pred_list_dict, y_true_list_dict, mixup_label_indices=mixup_label_indexes, mixup_lambda=mixup_lambda_values)
        
        # store this epoch's AUC and loss values
        train_avg_auc_values_list.append(train_avg_auc_value)
        train_loss_values_list.append(train_loss_value_epoch)
        train_auc_values_list_dict = misc.append_to_list_dict(train_auc_values_list_dict, train_auc_value_dict)
        train_loss_list_dict = misc.append_to_list_dict(train_loss_list_dict, train_loss_dict)

        # Print the training stats of this epoch
        if cfg.loss_function_name in ["mse"]:
                train_auc_value_dict = train_loss_dict
                train_avg_auc_value = train_loss_value_epoch

        logger.add_epoch_dict(name="Train", auc_dict=train_auc_value_dict, loss=train_loss_value_epoch, 
                              avg_auc=train_avg_auc_value, epoch_n=epoch)

        ###  3. validate the model (internal validation set)   ###
        if epoch % cfg.eval_interval == 0:
            val_loss_value, val_loss_dict, val_auc_value_dict, val_avg_auc_value, _, _, _ = validate(cfg=cfg, model=model, dataloader=val_dataloader,
                                                                            mode='internal validation', loss_function=loss_function, logger=logger, save_outputs=False)
            
            if cfg.loss_function_name in ["mse"]:
                val_auc_value_dict = val_loss_dict
                val_avg_auc_value = val_loss_value

            logger.add_epoch_dict(name="Validation", auc_dict=val_auc_value_dict, loss=val_loss_value, avg_auc=val_avg_auc_value, epoch_n=epoch)

            # Save validation results
            val_loss_values_list.append(val_loss_value)
            val_avg_auc_values_list.append(val_avg_auc_value)
            val_auc_values_list_dict = misc.append_to_list_dict(val_auc_values_list_dict, val_auc_value_dict)
            val_loss_list_dict = misc.append_to_list_dict(val_loss_list_dict, val_loss_dict)

            # # DANIEL: EARLY STOPPING AND LAYER FREEZING
            if cfg.use_layer_freezing:
                STOP_TRAINING, model, optimizer, scheduler, logger, \
                best_metrics_tracker, model_backbone_frozen = layer_freezing(cfg, model, optimizer, scheduler, best_metrics_tracker, model_backbone_frozen, \
                                                                                    epoch, val_loss_dict, val_loss_value, \
                                                                            val_avg_auc_value, val_auc_value_dict, logger)
            # normal early stopping                                                                                  
            else:
                STOP_TRAINING, model, optimizer, scheduler, logger, \
                best_metrics_tracker = early_stopping(cfg, model, optimizer, scheduler, best_metrics_tracker, 
                                                      logger, epoch, val_loss_value, val_loss_dict, val_avg_auc_value, val_auc_value_dict)
                
            if STOP_TRAINING:
                break

    
    
    if STOP_TRAINING:
        # if the boolean was used to trigger the end of training
        logger.my_print(f"Loss has not improved for {cfg.patience}: STOPPING TRAINING")
    else:
        # this statement is only reached if the max_epochs is reached
        logger.my_print(f'Maximum number of epochs ({max_epochs}) reached: STOPPING TRAINING.')
    
    # go back and load the best model
    if os.path.exists(cfg.temp_model_weights_dir) and cfg.use_layer_freezing:
        model.load_state_dict(torch.load(os.path.join(cfg.exp_dir, cfg.filename_best_model_pth)))

        model = load_model_output_head_weights(config=cfg, model=model, 
                                                           output_heads_to_load=cfg.endpoint_list, freeze_loaded_heads=True)
        # when finished training delete the temporary folder of saved output heads weights
        shutil.rmtree(cfg.temp_model_weights_dir)
    
        torch.save(model.state_dict(), os.path.join(cfg.exp_dir, cfg.filename_best_model_pth))

    best_val_epoch_num = best_metrics_tracker.best_val_epoch_num

    return (train_loss_values_list, train_loss_list_dict, train_avg_auc_values_list, train_auc_values_list_dict,  
            val_loss_values_list, val_loss_list_dict,  val_avg_auc_values_list,   val_auc_values_list_dict,
            best_val_epoch_num, lr_list, batch_size_list, epoch, GPU_RAM_USAGE)
        






def validate(cfg, model, dataloader, mode, loss_function, logger, save_outputs=True):
    """
    Validate the model.

    Args:
        cfg: the config object
        model: the DL model
        dataloader: the dataloader of the dataset (train, val, or test) to test the model on.
        mode (str): for printing, either 'internal validation' or 'test'
        loss_function: 
        logger:
        save_outputs: boolean, whether or not to save the model predictions on this set

        loss_value: the mean loss on this set
        loss_dict: a dictionary containing the mean loss per endpoint
        auc_value_dict: the AUC value for each endpoint
        avg_auc_value: the average AUC over all endpoints
        patient_ids: a list of patient IDs in this set
        y_pred_list_dict: the model's predictions on this set (a dictionary of lists, key=endpoint)
        y_true_list_dict: the true labels for the patients (also a dictinary of lists, same structure)
    """
    endpoint_list = cfg.endpoint_list
    device = cfg.device

    # Initialize variable
    model.eval()
    mode = mode.capitalize()
    loss_value = 0
    loss_dict = {endpoint: 0 for endpoint in endpoint_list}
    num_iterations = len(dataloader)
    patient_ids = list()
    valid_endpoints_as_tensor = torch.as_tensor(cfg.valid_endpoint_values, device=device)

    with torch.no_grad():
        model.eval()
        y_pred_dict = dict()
        y_true_dict = dict()
        for endpoint in endpoint_list:
            y_pred_dict[endpoint] = torch.as_tensor([], dtype=torch.float32, device=device)
            y_true_dict[endpoint] = torch.as_tensor([], dtype=torch.int8, device=device)

        for data in dataloader:
            # Load data
            inputs, features, label_list = (
                data['ct_dose_seg'].to(device),
                data['features'].to(device),
                data['label_list'].to(device),
                )
            label_dict = dict()
            for i, (endpoint, label) in enumerate(zip(endpoint_list, torch.swapaxes(label_list, 0, 1))):
                label_dict[endpoint] = label

            # Model make predictions
            outputs_dict = model(x=inputs, features=features)

            # Calculate loss
            loss = 0
            for endpoint in endpoint_list:
                # save ALL predictions on this epoch (including ones for missing labels)
                y_pred_dict[endpoint] = torch.cat([y_pred_dict[endpoint], outputs_dict[endpoint]], dim=0)
                y_true_dict[endpoint] = torch.cat([y_true_dict[endpoint], label_dict[endpoint]], dim=0)

            batch_loss, batch_per_toxicity_loss_dict, \
            unweighted_batch_loss_mean, unweighted_batch_per_toxicity_loss_dict = losses.calculate_loss(cfg, outputs_dict, label_dict, valid_endpoints_as_tensor, loss_function,
                                                                                                        mixup_y_true_indexes=None, mixup_lambda=None)
            
            # collect the losses of this epoch (divided by the number of iterations)
            loss_value += unweighted_batch_loss_mean.item() / num_iterations
            loss_dict = misc.sum_dictionaries(loss_dict, unweighted_batch_per_toxicity_loss_dict, divisor=num_iterations)

            # Save patient_ids
            patient_ids += data['patient_id']

            del inputs, features, loss, outputs_dict, data, batch_loss, batch_per_toxicity_loss_dict, unweighted_batch_loss_mean, unweighted_batch_per_toxicity_loss_dict

        # Compute internal validation AUC (using the non-missing endpoint values only)
        y_pred_list_dict, y_true_list_dict = misc.apply_sigmoid_or_softmax(cfg, y_pred_dict, y_true_dict)
        avg_auc_value, auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, y_pred_list_dict, y_true_list_dict)

        # Save test set outputs to csv
        if mode.lower() == 'test' and save_outputs:
            mode_list = ['test'] * len(patient_ids)
            misc.save_predictions(config=cfg, patient_ids=patient_ids, endpoint_list=endpoint_list,
                                    y_pred_list_dict=y_pred_list_dict, y_true_list_dict=y_true_list_dict,
                                    mode_list=mode_list, model_name=cfg.model_name,
                                    exp_dir=cfg.exp_dir, logger=logger, external_set = False)

        return loss_value, loss_dict, auc_value_dict, avg_auc_value, patient_ids, y_pred_list_dict, y_true_list_dict




def perform_N_fold_cross_validation(cfg, main_logger, hyperparameter_tuning_mode=False):
    """
    Performs N-fold cross-validation on the model. If `cfg.cv_folds` is 1, then it just one train/val split is used. 
    Args:
        cfg: the config object
        logger: the logger object
        hyperparameter_tuning_mode: whether or not this is a hyperparameter tuning run. If `True`, then many of the CLI 
                                logging statements are suppressed (but they are still saved to the log.txt files)
    Returns:
        
    """

    # Set random seeds (sets a new one if we're doing HP tuning)
    cfg = misc.set_random_seeds(cfg, set_new_random_seed=hyperparameter_tuning_mode)
    cv_folds = cfg.cv_folds
    
    train_aucs_list_dict = {endpoint: list() for endpoint in cfg.endpoint_list}
    val_aucs_list_dict   = {endpoint: list() for endpoint in cfg.endpoint_list}
    test_aucs_list_dict  = {endpoint: list() for endpoint in cfg.endpoint_list}
    ensemble_DL_predictions = {endpoint: None for endpoint in cfg.endpoint_list}
    train_losses_list, val_losses_list, test_losses_list = [], [], []
    lr_train_aucs_list_dict = {endpoint: list() for endpoint in cfg.endpoint_list}
    lr_val_aucs_list_dict   = {endpoint: list() for endpoint in cfg.endpoint_list}
    lr_test_aucs_list_dict  = {endpoint: list() for endpoint in cfg.endpoint_list}
    ensemble_LR_predictions = {endpoint: None for endpoint in cfg.endpoint_list}

    DL_val_ECE, DL_val_MCE, LR_val_ECE, LR_val_MCE = None, None, None, None
    DL_test_ECE, DL_test_MCE, LR_test_ECE, LR_test_MCE = None, None, None, None
    DL_ens_ECE, DL_ens_MCE, LR_ens_ECE, LR_ens_MCE = None, None, None, None

    DL_train_ECE_list_dict, DL_val_ECE_list_dict, DL_test_ECE_list_dict = {endpoint: list() for endpoint in cfg.endpoint_list}, {endpoint: list() for endpoint in cfg.endpoint_list}, {endpoint: list() for endpoint in cfg.endpoint_list}
    #DL_train_MCE_list, DL_val_MCE_list, DL_test_MCE_list = [], [], []
    LR_train_ECE_list_dict, LR_val_ECE_list_dict, LR_test_ECE_list_dict = {endpoint: list() for endpoint in cfg.endpoint_list}, {endpoint: list() for endpoint in cfg.endpoint_list}, {endpoint: list() for endpoint in cfg.endpoint_list}
    #LR_train_MCE_list, LR_val_MCE_list, LR_test_MCE_list = [], [], []

    if cv_folds > 1:
        # Fetch training, internal validation and test files
        cv_dict_0, cv_dict_1, test_dict = load_data.get_files(config=cfg, logger=main_logger)
        # Combine training and internal validation files into one set (which we can do cross-val folds with)
        cv_dict = cv_dict_0 + cv_dict_1

        # Determine endpoint values to stratify on
        cv_y = [x['label_list'] for x in cv_dict]
        cv_y = LabelEncoder().fit_transform([''.join(str(l)) for l in cv_y])  # turn each multi-label list into a sting (e.g. [0-1110-1] for [0, -1, 1, 1, 0, -1])

        # Create stratified CV class
        if cfg.cv_type == 'stratified':
            main_logger.my_print(f'Performing stratified {cfg.cv_folds}-fold CV.')
            cv_object = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.seed) 
        else:
            main_logger.my_print(f'Performing randomised {cv_folds}-fold CV.')
            cv_object = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.seed)

        # For each fold: determine train and val indices
        cv_idx_list = list()
        for idx in cv_object.split(X=cv_dict, y=cv_y):
            cv_idx_list.append(idx)
    
    # Run cross validation folds
    for fold_idx in range(cv_folds):
        fold_n = fold_idx + 1  # start indexing at 1, for the print statements
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # clear cache before each fold (try to free up some GPU space)
        logger = initialize(config=cfg, device=cfg.device, create_folders=True)

        if cv_folds > 1:
            # For this fold: select training and internal validation indices
            train_idx, valid_idx = cv_idx_list[fold_idx]
            train_dict = [cv_dict[i] for i in train_idx]
            val_dict = [cv_dict[i] for i in valid_idx]
            assert len(train_dict) + len(val_dict) == len(cv_dict)
        elif cv_folds == 1: # just use 1 train and 1 val dict
            train_dict, val_dict, test_dict = load_data.get_files(config=cfg, logger=logger)
            
        if cfg.use_test_set == False:  # prevents loading the test set at all
            test_dict = None

        # Get the datloaders for this fold 
        (train_dataloader, val_dataloader, test_dataloader, batch_size, channels, depth, height, width, 
         n_features, train_dict, val_dict, test_dict) = \
            load_data.main(config=cfg, train_dict=train_dict, val_dict=val_dict, test_dict=test_dict,
                           fold=fold_n, drop_last=cfg.dataloader_drop_last, logger=logger)

        # Initialize model (weights are also initialized via this function)
        if cfg.pretrained_path is not None:
            #pretrained_path_i = os.path.join(cfg.pretrained_path, str(fold_n), cfg.filename_best_model_pth)
            pretrained_path_i = os.path.join(cfg.pretrained_path, cfg.filename_best_model_pth)
        else:
            pretrained_path_i = None
        
        try:
            model, model_total_params = get_classification_model(config=cfg, channels=channels, depth=depth, height=height, width=width, n_features=n_features,
                                    pretrained_path=pretrained_path_i, logger=logger)
            model.to(device=cfg.device)
        except Exception as e:
            logger.my_print("Out of memory error. Skipping trial.")
            if cfg.hyperparameter_tuning_mode:
                raise e
                report_invalid_model_trial(config=cfg)
            else:
                raise e
 

        # print which fold we're on (slightly neater to print this after the model summary)
        logger.my_print(f'FOLD {fold_n}/{cv_folds}...')
        
        # Compile model (PyTorch 2)  #os.name = 'nt' is windows, torch.compile is not supported on windows
        if cfg.torch_version.startswith('2.') and os.name != 'nt' and cfg.device != (torch.device('cpu') or 'cpu'):
            #model = torch.compile(model, mode='default')
            num_GPUs = torch.cuda.device_count()
            logger.my_print(f'Model compiled! Running on {num_GPUs} GPUs')
        else:
            logger.my_print('Windows OR no GPU = No model compiling.')

        # get the loss function here
        loss_function = losses.get_loss_function(config=cfg, train_dict=train_dict, val_dict=val_dict)

        #if :  # Just run a pretrained model on the datasets (i.e. do not train it)
        if cfg.test_model_mode:
            # have to give these variables a value, or else they will crash the results-saving code
            epoch = "None"
            best_val_epoch_num = "Eval"
            lr_list = "None"
            batch_size_list= "None"

        if not cfg.test_model_mode:  # Train DL model     
            (train_loss_values_list, train_loss_list_dict, train_avg_auc_values_list, train_auc_values_list_dict,  
            val_loss_values_list,  val_loss_list_dict,  val_avg_auc_values_list,   val_auc_values_list_dict,
            best_val_epoch_num, lr_list, batch_size_list, epoch, GPU_RAM_USAGE) = \
                train_DL_model(cfg=cfg, model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                            loss_function=loss_function, fold_n = fold_n, logger=logger)
            
            # plot the train and val losses
            misc.plot_losses_dicts(cfg, train_loss_list_dict, val_loss_list_dict, train_loss_values_list, val_loss_values_list)

            # Plot training and internal validation losses
            y_list = [[train_loss_values_list, val_loss_values_list]] +\
                        [[train_avg_auc_values_list, val_avg_auc_values_list]] + \
                        [[train_auc_values_list_dict[endpoint], val_auc_values_list_dict[endpoint]] for endpoint in cfg.endpoint_list] + \
                        [[lr_list], [GPU_RAM_USAGE]]
            y_label_list = ['Loss', 'Average AUC'] + [f'{x} AUC' for x in cfg.endpoint_list] + ['LR', 'GPU']
        
            freeze_epoch_list = [None, None] + [None for endpoint in cfg.endpoint_list] + [None, None]                # TODO: this is temporary !!
            best_epoch_list = [None, best_val_epoch_num] + [None for endpoint in cfg.endpoint_list] + [None, None]    # TODO: so is this...
            legend_list = [['Training', 'Internal validation']] * (len(y_list) - 2) + [None, None]
            misc.plot_values(y_list=y_list, y_label_list=y_label_list, best_epoch_list=best_epoch_list,
                                freeze_epoch_list=freeze_epoch_list, legend_list=legend_list,
                                figsize=cfg.figsize, save_filename=os.path.join(cfg.exp_dir, cfg.filename_results_png))
            
            if cfg.device == torch.device('cuda'):
                logger.my_print(f"Mean GPU usage: {np.mean(GPU_RAM_USAGE):.2f} GB")
                logger.my_print(f"Max GPU usage: {np.max(GPU_RAM_USAGE):.2f} GB")
                mem_free, mem_toal = torch.cuda.mem_get_info()
                logger.my_print(f"Free GPU: {mem_free/1e9:.2f} GB")
                logger.my_print(f"Total GPU: {mem_toal/1e9:.2f} GB")

            # Test the best model from this fold, on the train, val and test sets
            model.load_state_dict(torch.load(os.path.join(cfg.exp_dir, cfg.filename_best_model_pth)))
            model.eval()

        logger.my_print("Testing the best model from this fold on the training and validation sets")
        train_loss_value, train_loss_dict, train_auc_value_dict, train_avg_auc_value, train_patient_ids, train_y_pred_list_dict, train_y_true_list_dict = \
                validate(cfg=cfg, model=model, dataloader=train_dataloader, 
                         mode='test', loss_function=loss_function, logger=logger, save_outputs=False)
        val_loss_value, val_loss_dict, val_auc_value_dict, val_avg_auc_value, val_patient_ids, val_y_pred_list_dict, val_y_true_list_dict =\
                validate(cfg=cfg, model=model, dataloader=val_dataloader, 
                         mode='test', loss_function=loss_function, logger=logger, save_outputs=False)
        logger.add_epoch_dict(name="Train Set", auc_dict=train_auc_value_dict, loss=train_loss_value, avg_auc=train_avg_auc_value, epoch_n=epoch)
        logger.add_epoch_dict(name="Validation Set", auc_dict=val_auc_value_dict, loss=val_loss_value, avg_auc=val_avg_auc_value, epoch_n=epoch)
        
        all_patient_ids = train_patient_ids + val_patient_ids
        all_modes = ['train'] * len(train_patient_ids) + ['val'] * len(val_patient_ids)
        
        # test the best model from this fold on the test set
        if cfg.use_test_set:
            # (this function also already saves the predictions to csv)
            logger.my_print("Testing the best model from this fold on the TESTING set")
            test_loss_value, test_loss_dict, test_auc_value_dict, test_avg_auc_value, test_patient_ids, test_y_pred_list_dict, test_y_true_list_dict = \
                validate(cfg=cfg, model=model, dataloader=test_dataloader, 
                         mode='test', loss_function=loss_function, logger=logger, save_outputs=True)
            
            logger.add_epoch_dict(name="Test Set", auc_dict=test_auc_value_dict, loss=test_loss_value, avg_auc=test_avg_auc_value, epoch_n=epoch)
            
            # Ensemble predictions on the testing set
            if cfg.cv_folds > 1:
                # add this fold's predictions to the ensemble predictions
                for endpoint in cfg.endpoint_list:
                    if ensemble_DL_predictions[endpoint] is None:
                        ensemble_DL_predictions[endpoint] = [x for x in test_y_pred_list_dict[endpoint]]
                    else:
                        ensemble_DL_predictions[endpoint] = [x+y for x,y in zip(ensemble_DL_predictions[endpoint], test_y_pred_list_dict[endpoint])]

                # on last fold, calculate ensemble AUCs and save the predictions
                if (fold_n == cfg.cv_folds):
                    for endpoint in cfg.endpoint_list:
                        ensemble_DL_predictions[endpoint] = [x/cfg.cv_folds for x in ensemble_DL_predictions[endpoint]]

                    ens_avg_auc_value, ens_auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, ensemble_DL_predictions, test_y_true_list_dict)

                    logger.add_epoch_dict(name="Ensemble test", auc_dict=ens_auc_value_dict, loss='N/A', avg_auc=ens_avg_auc_value, epoch_n='N/A')

                    mode_list = ['test'] * len(test_patient_ids)
                    misc.save_predictions(config=cfg, patient_ids=test_patient_ids, endpoint_list=cfg.endpoint_list,
                                    y_pred_list_dict=ensemble_DL_predictions, y_true_list_dict=test_y_true_list_dict,  # test_y_true_list_dict should be the same for all folds!
                                    mode_list=mode_list, model_name=cfg.model_name + '_ens',
                                    exp_dir=cfg.exp_dir, logger=logger) 

            # join test predictions to the whole set of predictions
            all_patient_ids += test_patient_ids
            all_modes += ['test'] * len(test_patient_ids)
            
        else: # not using the testing set
            test_loss_value, test_auc_value_dict, test_avg_auc_value = None, {endpoint : None for endpoint in cfg.endpoint_list}, None
            test_y_pred_list_dict = {endpoint : [] for endpoint in cfg.endpoint_list}
            test_y_true_list_dict = {endpoint : [] for endpoint in cfg.endpoint_list}

        # print the results of the best model of this fold
        logger.print_epoch_table(mode="Final Model")

        # join the train-val-test sets predictions and AUC values together
        all_y_pred_list_dict, all_y_true_list_dict = misc.concatenate_predictions(cfg, train_y_pred_list_dict, train_y_true_list_dict, 
                                                                                val_y_pred_list_dict, val_y_true_list_dict, 
                                                                                test_y_pred_list_dict, test_y_true_list_dict)
        # save train, validation and test set predictions to csv
        misc.save_predictions(config=cfg, patient_ids=all_patient_ids, endpoint_list=cfg.endpoint_list,
                                  y_pred_list_dict=all_y_pred_list_dict, y_true_list_dict=all_y_true_list_dict,
                                  mode_list=all_modes, model_name=cfg.model_name + '_all',
                                  exp_dir=cfg.exp_dir, logger=logger)
        
        [train_aucs_list_dict, val_aucs_list_dict, test_aucs_list_dict] = \
            misc.append_to_list_dicts(config=cfg, list_dicts = [train_aucs_list_dict, val_aucs_list_dict, test_aucs_list_dict], 
                                   value_dicts = [train_auc_value_dict, val_auc_value_dict, test_auc_value_dict])
        
        train_losses_list.append(train_loss_value)
        val_losses_list.append(val_loss_value)
        if cfg.use_test_set: test_losses_list.append(test_loss_value)


        """
        Logistic Regression Models of this fold
        """
        if cfg.train_CITOR_models:
            logger.my_print("Fitting Logistic Regression Models...")
            (train_patient_ids_lr, train_y_pred_lr_list_dict, train_y_lr_list_dict,
            val_patient_ids_lr, val_y_pred_lr_list_dict, val_y_lr_list_dict,
            test_patient_ids_lr, test_y_pred_lr_list_dict, test_y_lr_list_dict, lr_coefficients) = \
                refit_logistic_regression(config=cfg, patient_id_col=data_preproc_cfg.patient_id_col, logger=logger)
            
            all_lr_patient_ids = train_patient_ids_lr + val_patient_ids_lr 
            all_lr_modes = ['train'] * len(train_patient_ids_lr) + ['val'] * len(val_patient_ids_lr) 
            all_lr_y_pred_list_dict, all_lr_y_true_list_dict = misc.concatenate_predictions(cfg, 
                                                                                            train_y_pred_lr_list_dict, train_y_lr_list_dict, 
                                                                                            val_y_pred_lr_list_dict, val_y_lr_list_dict, 
                                                                                            test_y_pred_lr_list_dict, test_y_lr_list_dict)

            train_lr_avg_auc_value, train_lr_auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, train_y_pred_lr_list_dict, train_y_lr_list_dict)
            val_lr_avg_auc_value, val_lr_auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, val_y_pred_lr_list_dict, val_y_lr_list_dict)

            logger.my_print("Logistic Regression Model Results:")
            logger.my_print(f"Mean Train AUC: {train_lr_avg_auc_value:.3f}. AUCs per endpoint: {train_lr_auc_value_dict}")
            logger.my_print(f"Mean Val AUC:   {val_lr_avg_auc_value:.3f}. AUCs per endpoint: {val_lr_auc_value_dict}")

            if cfg.use_test_set:
                test_lr_avg_auc_value, test_lr_auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, test_y_pred_lr_list_dict, test_y_lr_list_dict)
                logger.my_print(f"Mean Test AUC:  {test_lr_avg_auc_value:.3f}. AUC per endpoint: {test_lr_auc_value_dict}")
                all_lr_patient_ids += test_patient_ids_lr
                all_lr_modes += ['test'] * len(test_patient_ids_lr)
                # Ensemble predictions on the testing set
                if cfg.cv_folds > 1:
                    # add this fold's predictions to the ensemble predictions
                    for endpoint in cfg.endpoint_list:
                        if ensemble_LR_predictions[endpoint] is None:
                            ensemble_LR_predictions[endpoint] = [x for x in test_y_pred_lr_list_dict[endpoint]]
                        else:
                            ensemble_LR_predictions[endpoint] = [x+y for x,y in zip(ensemble_LR_predictions[endpoint], test_y_pred_lr_list_dict[endpoint])]
                        

                    # on last fold, calculate ensemble AUCs and save the predictions
                    if (fold_n == cfg.cv_folds):
                        for endpoint in cfg.endpoint_list:
                            ensemble_LR_predictions[endpoint] = [x/cfg.cv_folds for x in ensemble_LR_predictions[endpoint]]
                            #print(len(ensemble_LR_predictions[endpoint]))
                        lr_ens_avg_auc_value, lr_ens_auc_value_dict = metrics.compute_AUCs_for_multiple_endpoints(cfg, ensemble_LR_predictions, test_y_lr_list_dict)

                        logger.add_epoch_dict(name="Ensemble test", auc_dict=ens_auc_value_dict, loss='N/A', avg_auc=ens_avg_auc_value, epoch_n='N/A')

                        mode_list = ['test'] * len(test_patient_ids_lr)
                        misc.save_predictions(config=cfg, patient_ids=test_patient_ids_lr, endpoint_list=cfg.endpoint_list,
                                        y_pred_list_dict=ensemble_LR_predictions, y_true_list_dict=test_y_lr_list_dict,  # test_y_true_list_dict should be the same for all folds!
                                        mode_list=mode_list, model_name='lr_ens',
                                        exp_dir=cfg.exp_dir, logger=logger) 

                        # make a calibration plot for the ensemble predictions
                        ensemble_calibration_plot_dict_list = [
                            {
                                "name" : "DL Model",
                                "preds" : ensemble_DL_predictions,
                                "labels" : test_y_true_list_dict,
                            },{
                                "name" : "CITOR",
                                "preds" : ensemble_LR_predictions,
                                "labels" : test_y_lr_list_dict,
                            }
                        ]
                        adaptive_make_calibration_plots(cfg, ensemble_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='calibration', 
                                                        title="Calibration Plot: Ensemble (Test Set)", filedir = os.path.join(cfg.exp_dir, 'calibration_plot_ensemble.png'))
                        [DL_ens_ECE, LR_ens_ECE], [DL_ens_MCE, LR_ens_MCE] = \
                            adaptive_make_calibration_plots(cfg, ensemble_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='reliability',
                                                             title="Reliability Plot: Ensemble (Test Set)", filedir = os.path.join(cfg.exp_dir, 'reliability_plot_ensemble.png'))
        

            else:
                test_lr_avg_auc_value = None
                test_lr_auc_value_dict = {endpoint: [] for endpoint in cfg.endpoint_list}

            [lr_train_aucs_list_dict, lr_val_aucs_list_dict, lr_test_aucs_list_dict] = \
                misc.append_to_list_dicts(config=cfg, list_dicts = [lr_train_aucs_list_dict, lr_val_aucs_list_dict, lr_test_aucs_list_dict], 
                                    value_dicts = [train_lr_auc_value_dict, val_lr_auc_value_dict, test_lr_auc_value_dict])
            
            misc.save_predictions(config=cfg, patient_ids=all_lr_patient_ids, endpoint_list=cfg.endpoint_list,
                                  y_pred_list_dict=all_lr_y_pred_list_dict, y_true_list_dict=all_lr_y_true_list_dict,
                                  mode_list=all_lr_modes, model_name='_lr_all',
                                  exp_dir=cfg.exp_dir, logger=logger)

        # No CITOR models
        else:
            val_y_pred_lr_list_dict = None
            val_y_lr_list_dict = None
            test_y_pred_lr_list_dict = None
            test_y_lr_list_dict = None

        """
        Calibration Plots
        """
        val_calibration_plot_dict_list = [
            {
                "name" : "DL Model",
                "preds" : val_y_pred_list_dict,
                "labels" : val_y_true_list_dict,
            },{
                "name" : "CITOR",
                "preds" : val_y_pred_lr_list_dict,
                "labels" : val_y_lr_list_dict,
            }
        ]
        adaptive_make_calibration_plots(cfg, val_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='calibration', 
                                        title="Calibration Plot: Validation Set", filedir = os.path.join(cfg.exp_dir, 'calibration_plot_validation.png'))
        [DL_val_ECE, LR_val_ECE], [DL_val_MCE, LR_val_MCE] = \
            adaptive_make_calibration_plots(cfg, val_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='reliability',
                                          title="Reliability Plot: Validation Set", filedir = os.path.join(cfg.exp_dir, 'reliability_plot_validation.png'))
        
        
        [DL_val_ECE_list_dict, LR_val_ECE_list_dict] = \
                misc.append_to_list_dicts(config=cfg, list_dicts =[DL_val_ECE_list_dict, LR_val_ECE_list_dict], 
                                    value_dicts = [DL_val_ECE, LR_val_ECE])
        

        # also for test set (if it was used)
        if cfg.use_test_set:  
            test_calibration_plot_dict_list = [
                {
                    "name" : "DL Model",
                    "preds" : test_y_pred_list_dict,
                    "labels" : test_y_true_list_dict,
                },{
                    "name" : "CITOR",
                    "preds" : test_y_pred_lr_list_dict,
                    "labels" : test_y_lr_list_dict,
                }
            ]
            adaptive_make_calibration_plots(cfg, test_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='calibration', 
                                            title="Calibration Plot: Test Set", filedir = os.path.join(cfg.exp_dir, 'calibration_plot_test.png'))
            [DL_test_ECE, LR_test_ECE], [DL_test_MCE, LR_test_MCE] = \
                adaptive_make_calibration_plots(cfg, test_calibration_plot_dict_list, column_names = cfg.endpoint_list, mode='reliability',
                                                title="Reliability Plot: Test Set", filedir = os.path.join(cfg.exp_dir, 'reliability_plot_test.png'))
            
        """
        Saving results
        """
        
        
        # save all results
        results_kwargs_dict = {'seed': cfg.seed, 'train_loss': train_loss_value, 'val_loss': val_loss_value, 'test_loss_value': test_loss_value,
                    'train_avg_auc': train_avg_auc_value, 'val_avg_auc': val_avg_auc_value, 'test_avg_auc': test_avg_auc_value,
                    'train_aucs': train_auc_value_dict, 'val_aucs': val_auc_value_dict, 'test_aucs': test_auc_value_dict,
                    'DL_val_ECE': DL_val_ECE, 'DL_val_MCE': DL_val_MCE, 
                    'DL_test_ECE': DL_test_ECE, 'DL_test_MCE': DL_test_MCE, 
                    
                    'batch_size': batch_size_list, 'epochs': epoch, 'max_epochs': cfg.max_epochs, 'best_val_epoch': best_val_epoch_num,
                    'cv_folds': cfg.cv_folds, 'current_fold': fold_n,
                    # about the model:
                    'model_name': cfg.model_name, 'total_params': model_total_params,
                    'optimizer_name': cfg.optimizer_name, 'scheduler_name': cfg.scheduler_name,
                    'loss_function_name': cfg.loss_function_name,
                    # some config stuff to save
                    'seg_orig_labels': cfg.seg_orig_labels, 'seg_target_labels': cfg.seg_target_labels,
                    'features_dl': cfg.features_dl
                            }
        
        if cfg.train_CITOR_models:
            LR_results_kwargs_dict = {
                        'lr_train_avg_auc': train_lr_avg_auc_value, 'lr_train_auc': train_lr_auc_value_dict,
                        'lr_val_avg_auc': val_lr_avg_auc_value, 'lr_val_auc': val_lr_auc_value_dict,
                        'lr_test_avg_auc': test_lr_avg_auc_value, 'lr_test_auc': test_lr_auc_value_dict,
                        'LR_val_ECE': LR_val_ECE, 'LR_val_MCE': LR_val_MCE,
                        'LR_test_ECE': LR_test_ECE, 'LR_test_MCE': LR_test_MCE,

                        'lr': lr_list,
                        'lr_coefficients': lr_coefficients
                        }
        
            results_kwargs_dict.update(LR_results_kwargs_dict)

        misc.create_results(cfg, **results_kwargs_dict)

        # Rename folder
        src_folder_name = os.path.join(cfg.exp_dir)
        dst_folder_name = os.path.join(cfg.exp_dir +
                                       '_{}'.format(int(fold_n)) +
                                       '_{}'.format(cfg.seed) +
                                       '_{}'.format(best_val_epoch_num) +
                                       '_{}'.format(cfg.model_name) +
                                       '_params_{}'.format(model_total_params) +
                                       '_auc' +
                                       '_tr_{}'.format(round(train_avg_auc_value, cfg.nr_of_decimals)) +
                                       #'_lr_{}'.format(round(np.mean([x for x in train_auc_lr_dict.values() if x is not None]), nr_of_decimals)) +
                                       '_val_{}'.format(round(val_avg_auc_value, cfg.nr_of_decimals))   #+
                                       #'_lr_{}'.format(round(np.mean([x for x in val_auc_lr_dict.values() if x is not None]), nr_of_decimals))
                                       )
        
        if cfg.use_test_set:
            dst_folder_name += '_test_{}'.format(round(test_avg_auc_value, cfg.nr_of_decimals))
            #dst_folder_name += '_lr_{}'.format(round(np.mean([x for x in test_auc_lr_dict.values() if x is not None]), nr_of_decimals))
        

        # compute statistics over all folds
        mean_train_auc_all_folds, mean_train_aucs_per_toxicity_all_folds = misc.summarise_results_over_folds(cfg, train_aucs_list_dict)
        mean_val_auc_all_folds,   mean_val_aucs_per_toxicity_all_folds   = misc.summarise_results_over_folds(cfg, val_aucs_list_dict)
        mean_test_auc_all_folds,  mean_test_aucs_per_toxicity_all_folds  = misc.summarise_results_over_folds(cfg, test_aucs_list_dict, test_set=True)

        lr_mean_train_auc_all_folds, lr_mean_train_aucs_per_toxicity_all_folds = misc.summarise_results_over_folds(cfg, lr_train_aucs_list_dict)
        lr_mean_val_auc_all_folds,   lr_mean_val_aucs_per_toxicity_all_folds   = misc.summarise_results_over_folds(cfg, lr_val_aucs_list_dict)
        lr_mean_test_auc_all_folds,  lr_mean_test_aucs_per_toxicity_all_folds  = misc.summarise_results_over_folds(cfg, lr_test_aucs_list_dict, test_set=True)

        DL_mean_val_ECE,  DL_mean_val_ECE_per_toxicity_all_folds  = misc.summarise_results_over_folds(cfg, DL_val_ECE_list_dict)
        DL_mean_test_ECE, DL_mean_test_ECE_per_toxicity_all_folds = misc.summarise_results_over_folds(cfg, DL_test_ECE_list_dict, test_set=True)
        LR_mean_val_ECE,  LR_mean_val_ECE_per_toxicity_all_folds  = misc.summarise_results_over_folds(cfg, LR_val_ECE_list_dict)
        LR_mean_test_ECE, LR_mean_test_ECE_per_toxicity_all_folds = misc.summarise_results_over_folds(cfg, LR_test_ECE_list_dict, test_set=True)

        mean_train_loss = np.mean(train_losses_list)
        mean_val_loss = np.mean(val_losses_list)
        mean_test_loss = np.mean(test_losses_list) if cfg.use_test_set else None


        if (fold_n == cfg.cv_folds):

            # add mean train and val AUCs to the folder name
            dst_folder_name += '_avg_tr_{}'.format(round(mean_train_auc_all_folds, cfg.nr_of_decimals))
            dst_folder_name += '_val_{}'.format(round(mean_val_auc_all_folds, cfg.nr_of_decimals))

            all_folds_results_kwargs_dict = {'folds':cfg.cv_folds, 'max_epochs' : cfg.max_epochs,
                                    # about the model:
                                    'model_name' : cfg.model_name, 'total_params' : model_total_params,
                                    'optimizer_name' : cfg.optimizer_name, 'scheduler_name' : cfg.scheduler_name,
                                    'loss_function_name' : cfg.loss_function_name,
                                    # results - mean statistics over the folds
                                    'mean_train_AUC' : mean_train_auc_all_folds, 'mean_val_AUC' : mean_val_auc_all_folds, 'mean_test_AUC' : mean_test_auc_all_folds,
                                    'mean_val_ECE' : DL_mean_val_ECE, 'mean_test_ECE' : DL_mean_test_ECE,
                                    'mean_toxicity_train_AUCs' : mean_train_aucs_per_toxicity_all_folds, 
                                    'mean_toxicity_val_AUCs' : mean_val_aucs_per_toxicity_all_folds, 
                                    'mean_toxicity_test_AUCs' : mean_test_aucs_per_toxicity_all_folds,
                                    'mean_toxicity_val_ECEs' : DL_mean_val_ECE_per_toxicity_all_folds,
                                    'mean_toxicity_test_ECEs' : DL_mean_test_ECE_per_toxicity_all_folds,

                                    'mean_train_loss' : mean_train_loss, 'mean_val_loss' : mean_val_loss, 'mean_test_loss' : mean_test_loss,

                                    'seg_orig_labels' : cfg.seg_orig_labels, 'seg_target_labels' : cfg.seg_target_labels,
                                    'features_dl' : cfg.features_dl
                                           }
            if cfg.train_CITOR_models:
                LR_all_folds_results_kwargs_dict = {'lr_mean_train_AUC' : lr_mean_train_auc_all_folds, 'lr_mean_val_AUC' : lr_mean_val_auc_all_folds, 'lr_mean_test_AUC' : lr_mean_test_auc_all_folds,
                                        'lr_mean_val_ECE' : LR_mean_val_ECE, 'lr_mean_test_ECE' : LR_mean_test_ECE,
                                        'lr_mean_toxicity_train_AUCs' : lr_mean_train_aucs_per_toxicity_all_folds,
                                        'lr_mean_toxicity_val_AUCs' : lr_mean_val_aucs_per_toxicity_all_folds,
                                        'lr_mean_toxicity_test_AUCs' : lr_mean_test_aucs_per_toxicity_all_folds,
                                        'lr_mean_toxicity_val_ECEs' : LR_mean_val_ECE_per_toxicity_all_folds,
                                        'lr_mean_toxicity_test_ECEs' : LR_mean_test_ECE_per_toxicity_all_folds,
                                        }
                all_folds_results_kwargs_dict.update(LR_all_folds_results_kwargs_dict)

            if cfg.cv_folds > 1 and cfg.use_test_set:
                # add to folder name
                dst_folder_name += '_test_{}'.format(round(mean_test_auc_all_folds, cfg.nr_of_decimals))
                dst_folder_name += '_ens_{}'.format(round(ens_avg_auc_value, cfg.nr_of_decimals))

                # add the ensemble results to the results dict
                DL_ens_results_dict = {'ens_avg_auc_value' : ens_avg_auc_value, 'ens_auc_value_dict' : ens_auc_value_dict,
                                    'DL_ens_ECE' : DL_ens_ECE, 'DL_ens_MCE' : DL_ens_MCE}
                all_folds_results_kwargs_dict.update(DL_ens_results_dict)
                
                if cfg.train_CITOR_models:
                    LR_ens_results_dict = {'lr_ens_avg_auc_value' : lr_ens_avg_auc_value, 'lr_ens_auc_value_dict' : lr_ens_auc_value_dict,
                                        'LR_ens_ECE' : LR_ens_ECE, 'LR_ens_MCE' : LR_ens_MCE,}
                    all_folds_results_kwargs_dict.update(LR_ens_results_dict)
                
            # save the results of all folds
            misc.create_ensemble_results(cfg, **all_folds_results_kwargs_dict)


        # delete the logger (a new on is made on each new fold)
        #if fold_n < cfg.cv_folds:
        logger.close()
        del logger
        del train_dataloader._iterator, val_dataloader._iterator
        del train_dataloader, val_dataloader
        model.cpu()
        del model
        if cfg.use_test_set:
            del test_dataloader._iterator, test_dataloader.dataset, test_dataloader
        clear_cache()
        
        if cfg.hyperparameter_tuning_mode:
            # delete the model, optimizer and scheduler files, to save space  (do this before renaming the file to save computation time) 
            os.remove(os.path.join(src_folder_name, cfg.filename_best_model_pth))  # they are still in the source folder (which hasn't been renamed yet)
            os.remove(os.path.join(src_folder_name, cfg.filename_best_optimizer_pth))  
            if cfg.scheduler_name is not None:
                os.remove(os.path.join(src_folder_name, cfg.filename_best_scheduler_pth))

        # rename the folder
        time.sleep(1)
        shutil.move(src_folder_name, dst_folder_name)
        time.sleep(1)
        if os.path.exists(src_folder_name) and os.path.isdir(src_folder_name):
            shutil.rmtree(src_folder_name)
        time.sleep(1)
        
        # Wrap up HP tuning
        if cfg.hyperparameter_tuning_mode:
            # on the last fold
            if fold_n == cfg.cv_folds:
                # report the mean losses and AUCs on train and validation sets
                report = {"done" : True,
                    "fold_n" : fold_n, "epochs" : 'done', "total_params" : model_total_params,
                        "train_loss" : mean_train_loss, "val_loss" : mean_val_loss,               
                        "mean_train_AUC" : mean_train_auc_all_folds, "mean_val_AUC" : mean_val_auc_all_folds}
                # also report the mean AUC for each endpoint over all folds, for both train and validation
                for endpoint in cfg.endpoint_list:
                    report[f"mean_{endpoint}_train_AUC"] = mean_train_aucs_per_toxicity_all_folds[endpoint]
                    report[f"mean_{endpoint}_val_AUC"] = mean_val_aucs_per_toxicity_all_folds[endpoint]
                    report[f"mean_{endpoint}_val_ECE"] = DL_mean_val_ECE_per_toxicity_all_folds[endpoint]
                train.report(report)
                
            # report when not on the last fold
            else:
                # report whenever a fold is done, but it isn't the last one
                report = {"done":False, "fold_n" : fold_n, "epochs" : 'done', 
                          "mean_train_AUC" : mean_train_auc_all_folds, "mean_val_AUC" : mean_val_auc_all_folds}
                for endpoint in cfg.endpoint_list:
                    report[f"mean_{endpoint}_train_AUC"] = 0
                    report[f"mean_{endpoint}_val_AUC"] = 0
                    report[f"mean_{endpoint}_val_ECE"] = 0
                if fold_n >= 2 and mean_val_auc_all_folds <= 0.55:  # early stopping - if the first two folds are <=0.5 AUC, then stop
                    report["done"] = True
                    report[f"mean_{endpoint}_train_AUC"] = mean_train_aucs_per_toxicity_all_folds[endpoint]
                    report[f"mean_{endpoint}_val_AUC"] = mean_val_aucs_per_toxicity_all_folds[endpoint]
                    report[f"mean_{endpoint}_val_ECE"] = DL_mean_val_ECE_per_toxicity_all_folds[endpoint]
                train.report(report)






def ray_tune_model_hyperparameters(logger): 
    """
    This function controls the model hyperparameter tuning process. It uses Ray Tuning to do this.
    It is able to pick up where it left off on a previous run, and it can also resume a failed run.
    Tuning is done by using cross validation on each set of hyperparameters that the search algorithm suggests.

    Args:
        logger:
    Returns:
        None
    """
    # make it possible to pick up where we left off on a previous run
    tune_experiment_dir = hp_cfg.tune_experiment_dir
    storage_path = os.path.split(tune_experiment_dir)[0]
    #storage_path = pathlib.PureWindowsPath(storage_path).as_posix()
    
    ray.shutdown()
    # suppress some of the logging during ray tune
    context = ray.init(log_to_driver=False)
    print(context.dashboard_url)
    
    num_CPUs_per_process = hp_cfg.num_CPUs_per_process
    num_GPUs_per_process = hp_cfg.num_GPUs_per_process
    logger.my_print(f"Number of CPUs per process: {num_CPUs_per_process}. Number of GPUs per process: {num_GPUs_per_process}.")
    
    # Ray Tune's algorithm
    sampler = optuna.samplers.TPESampler(n_startup_trials=hp_cfg.num_random_trials_start, seed=hp_cfg.seed, )
                                         #multivariate=True, constant_liar=True) 
    search_algorithm = OptunaSearch(space=hp_cfg.define_by_run_func, metric=hp_cfg.metrics, mode=hp_cfg.modes, sampler=sampler)  # use optuna
    # Concurrency limiter: prevents too many trials from running at the same time
    search_algorithm = ConcurrencyLimiter(search_algorithm, max_concurrent=hp_cfg.max_concurrent_processes)
    progress_reporter = TrialTerminationReporter(parameter_columns=['batch_size'], metric_columns = ["mean_val_AUC"], 
                                                 metric="mean_val_AUC", mode="max", sort_by_metric=True) # limit the number of columns printed by the progress reporter

    can_restart = tune.Tuner.can_restore(tune_experiment_dir)
    if can_restart == True:
        logger.my_print("RESUMING RAY TUNE EXPERIMENT")
        print("RESUMING RAY TUNE EXPERIMENT")
        try:
            #search_algorithm.restore_from_dir(tune_experiment_dir)
            #can_restart = False
            
            tuner = tune.Tuner.restore(tune_experiment_dir,
                                        trainable=tune.with_resources(HP_tune_trial, {"cpu": num_CPUs_per_process, "gpu": num_GPUs_per_process}),
                                        #param_space = hp_cfg.param_space,
                                        resume_errored = hp_cfg.resume_errored,
                                        restart_errored = hp_cfg.restart_errored,  # 
                                        resume_unfinished = hp_cfg.resume_unfinished,  # these two would be true, but I have not implemented checkpointing... so it will crash
                                        )
            
            
        except Exception as e: 
            logger.my_print(f"EXCEPTION: {e}", level='warning')
            can_restart = False # start a new trial if this one can't be recovered
    if can_restart == False:
        logger.my_print("\nSTARTING NEW RAY TUNE EXPERIMENT\n")
        print("\nSTARTING NEW RAY TUNE EXPERIMENT\n")
        tuner = tune.Tuner(
            tune.with_resources(HP_tune_trial, {"cpu": num_CPUs_per_process, "gpu": num_GPUs_per_process}),  # sets how many resources per trial
            run_config = train.RunConfig(
                storage_path=storage_path,
                name = hp_cfg.exp_name,
                progress_reporter = progress_reporter,
                callbacks = [BasicTrialLogger()],
                verbose = 1,
                #stop = stop_invald_trial,
                #checkpoint_config=CheckpointConfig(checkpoint_frequency=5)
                ),
            tune_config=tune.TuneConfig(
                reuse_actors = False, 
                search_alg = search_algorithm,
                num_samples = hp_cfg.num_HP_trials,             # This is how many total runs will be done
                trial_name_creator = trial_str_creator,
                trial_dirname_creator = trial_str_creator,
                ),
        )
    
    results = tuner.fit()
    df_results = results.get_dataframe()
    df_results.to_csv(os.path.join(tune_experiment_dir, f"{hp_cfg.exp_name}_total_results.csv"), sep=";", index=False)
    search_algorithm.save(os.path.join(tune_experiment_dir, "search_algorithm.pkl"))
    
    ray.shutdown()



def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # clear cache after each fold (try to free up some GPU space)



def HP_tune_trial(hp_param_config):
    """
    This function sets up one hyperparameter tuning trial, using Ray Tune.

    Args:
        hp_param_config: ray tune object (dictionary?) containing the settings for this trial
    Returns:
        
    """

    # clear some cache
    clear_cache()

    assert os.getcwd() == os.environ["TUNE_ORIG_WORKING_DIR"]

    # import the config file, and overwrite any hyperparameters that are being tuned
    import config as config_file

    cfg = make_config_object(config_file, hp_param_config=hp_param_config)

    # set the device that Ray Tune will use, and the experiment trial directories
    cfg.device = get_device()

    # new exp_dir should be within the ray tune trial folder
    new_exp_dir = os.path.join(ray.train.get_context().get_trial_dir(), 
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
    cfg = rename_folders_for_parameter_tuning(cfg, new_exp_dir=new_exp_dir)
    
    # make a new logger
    logger = initialize(config=cfg, device=cfg.device, create_folders=False)
    
    # begin N fold cross validation for this hypermarameter set
    perform_N_fold_cross_validation(cfg, main_logger=logger, hyperparameter_tuning_mode=True)

    # clear cache
    clear_cache()
    if os.path.exists(cfg.temp_model_weights_dir):
         # when finished training delete the temporary folder of saved output heads weights
        shutil.rmtree(cfg.temp_model_weights_dir)
    if os.path.exists(new_exp_dir):
        shutil.rmtree(new_exp_dir)

    
    



def main():
    """
    Main function. Controls what type of model training should occur 
    (e.g. hyperparameter tuning or regular 5 fold cross-validation).
    """

    import config as config_file
    cfg = make_config_object(config_file, hp_param_config=None)

    # Make sure that data_preproc_cfg.use_umcg == True
    assert data_preproc_cfg.use_umcg
    
    main_logger = initialize(config=cfg, device=cfg.device, create_folders=False)
    start_time = time.time()
    
    
    if cfg.hyperparameter_tuning_mode == True:
        ray_tune_model_hyperparameters(logger=main_logger)  # the testing set should never be used here. 

    elif cfg.subsample_train_set == True:
        for sample_size in cfg.subsample_train_set_sizes:
            
            new_exp_root_dir = os.path.join(cfg.root_path, 'experiments', 'Dataset_size', str(sample_size)) 
            cfg.exp_root_dir = new_exp_root_dir
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_exp_dir = os.path.join(new_exp_root_dir, exp_name)
  
            cfg.exp_dir = new_exp_dir
            cfg.subsample_train_set_size = sample_size

            cfg.exp_src_dir = os.path.join(new_exp_dir, 'src')
            cfg.exp_figures_dir = os.path.join(new_exp_dir, 'figures')
            cfg.temp_model_weights_dir = os.path.join(new_exp_dir, 'tmp_model_weights')
            # perform N-fold cross-validation on the subsampled dataset
            perform_N_fold_cross_validation(cfg=cfg, main_logger=main_logger)
    else:
        if cfg.cv_folds >= 1:
            main_logger.my_print(f"Performing {cfg.cv_folds}-fold cross-validation!")
            # train n models on n folds of CV
            perform_N_fold_cross_validation(cfg=cfg, main_logger=main_logger)

    
    end_time = time.time()
    main_logger.my_print('Elapsed time: {time} (H:M:S).'.format(time=time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))))






if __name__ == '__main__':     
    if not (os.name == 'nt'):
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
    # Initialize classes
    import config as config_file    

    misc.set_random_seeds(config_file, set_new_random_seed=False)

    misc.set_torch_reproduciblity_backends(config=config_file)
    
    main()
    
