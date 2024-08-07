"""
Script for developing deep learning models via cross-validation, ensembling the models (mean prediction), and
independently testing the final ensemble model.
"""
import os
import copy
import time
import torch
import shutil
import random
import numpy as np
import matplotlib

matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from monai.utils import set_determinism
from monai.metrics import ROCAUCMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
)

# Load local functions
import config
import misc
import utils.load_data as load_data
import process_data
import data_preproc.data_preproc_config as data_preproc_config

from data_preproc.data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from models.logistic_regression import run_logistic_regression

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.multiprocessing.set_sharing_strategy('file_system')

def initialize(torch_version, device, create_folders=True):
    """
    Set up experiment and save the experiment path to be able to save results.

    Args:
        torch_version:
        device:
        create_folders:

    Returns:

    """
    if create_folders:
        # Create experiment folder and subfolders if they do not exist yet
        exp_dir = config.exp_dir
        exp_src_dir = config.exp_src_dir
        exp_models_dir = config.exp_models_dir
        exp_optimizers_dir = config.exp_optimizers_dir
        exp_data_preproc_dir = config.exp_data_preproc_dir
        exp_figures_dir = config.exp_figures_dir
        for p in [exp_dir, exp_src_dir, exp_models_dir, exp_optimizers_dir, exp_data_preproc_dir, exp_figures_dir]:
            create_folder_if_not_exists(p)

        # Fetch folders and files that should be copied to exp folder
        root_path = config.root_path
        models_dir = config.models_dir
        optimizers_dir = config.optimizers_dir
        data_preproc_dir = config.data_preproc_dir
        src_files = [x for x in os.listdir(root_path) if
                     (x.endswith('.py') or x == data_preproc_config.filename_exclude_patients_csv)]

        # Copy src files to exp folder
        for f in src_files:
            copy_file(src=os.path.join(root_path, f), dst=os.path.join(exp_src_dir, f))
        # Copy folders to exp folder
        copy_folder(src=models_dir, dst=exp_models_dir)
        copy_folder(src=optimizers_dir, dst=exp_optimizers_dir)
        copy_folder(src=data_preproc_dir, dst=exp_data_preproc_dir)

        # Logger output filename
        output_filename = os.path.join(exp_dir, 'log.txt')
    else:
        output_filename = None

    # Initialize logger
    logger = Logger(output_filename=output_filename)

    # Print variables
    logger.my_print('Device: {}.'.format(device))
    logger.my_print('Torch version: {}.'.format(torch_version))
    logger.my_print('Torch.backends.cudnn.benchmark: {}.'.format(torch.backends.cudnn.benchmark))

    return logger


def validate(model, dataloader, mean, std, mode, logger, save_outputs=True):
    """
    Validate the model.

    Args:
        model:
        dataloader:
        mean: normalization: list of means, one value for each channel (first value for CT, second value for RTDOSE)
        std: normalization: list of stds, one value for each channel (first value for CT, second value for RTDOSE)
        mode (str): for printing, either 'internal validation' or 'test'
        logger:

    Returns:

    """
    # Initialize variable
    model.eval()
    mode = mode.capitalize()
    loss_value = 0
    num_iterations = len(dataloader)
    patient_ids = list()

    with torch.no_grad():
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

            # Preprocess inputs and features
            # NOTE: these operations should also be performed in lr_finder.py
            inputs = process_data.preprocess_inputs(inputs=inputs,
                                                    ct_mean=mean[0], ct_std=std[0],
                                                    rtdose_mean=mean[1], rtdose_std=std[1])
            features = process_data.preprocess_features(features=features)

            # Make predictions
            outputs_dict = model(x=inputs, features=features)

            # Calculate loss
            loss = 0
            nr_of_endpoints_with_loss = 0
            for endpoint in endpoint_list:
                # Only consider patients with non-missing value for `endpoint`, i.e., labels in `valid_endpoint_values`
                batch_size_endpoint_before = outputs_dict[endpoint].shape[0]
                outputs_dict[endpoint] = outputs_dict[endpoint][
                    torch.isin(label_dict[endpoint], torch.as_tensor(valid_endpoint_values, device=device))]
                label_dict[endpoint] = label_dict[endpoint][
                    torch.isin(label_dict[endpoint], torch.as_tensor(valid_endpoint_values, device=device))]

                outputs = outputs_dict[endpoint]
                labels = label_dict[endpoint]
                batch_size_endpoint_after = outputs_dict[endpoint].shape[0]

                if batch_size_endpoint_after > 0:
                    try:
                        # Cross-Entropy, Ranking, Custom
                        loss += loss_function(outputs, labels)  # * (batch_size_endpoint_after / batch_size_endpoint_before)
                    except:
                        # BCE
                        loss += loss_function(outputs, torch.reshape(labels, outputs.shape).to(outputs.dtype))   # * (batch_size_endpoint_after / batch_size_endpoint_before)
                    nr_of_endpoints_with_loss += 1

            # Scale loss value: has impact on weight updates! Else, the weights will be updated
            # `nr_of_endpoints_with_loss` times more aggresively than normal. Note: during validation, the weights
            # will NOT be updated, but we use the same calculation as during train(), for consistency (of plots)
            loss /= nr_of_endpoints_with_loss

            # Evaluate model (internal validation set)
            loss_value += loss.item()
            for endpoint in endpoint_list:
                y_pred_dict[endpoint] = torch.cat([y_pred_dict[endpoint], outputs_dict[endpoint]], dim=0)
                y_true_dict[endpoint] = torch.cat([y_true_dict[endpoint], label_dict[endpoint]], dim=0)

            # Save patient_ids
            patient_ids += data['patient_id']

        # Averaging internal validation loss
        loss_value /= num_iterations

        # Compute internal validation AUC
        y_pred_list_dict = dict()
        y_true_list_dict = dict()
        auc_value_dict = dict()
        for endpoint in endpoint_list:
            if loss_function_name in ['bce']:
                y_pred_list_dict[endpoint] = misc.proba(sigmoid_act(y_pred_dict[endpoint]))
            else:
                y_pred_list_dict[endpoint] = [softmax_act(i) for i in y_pred_dict[endpoint]]
            y_true_list_dict[endpoint] = [to_onehot(i) for i in y_true_dict[endpoint]]

            auc_value_dict[endpoint] = misc.compute_auc(y_pred_list=y_pred_list_dict[endpoint],
                                                        y_true_list=y_true_list_dict[endpoint],
                                                        auc_metric=auc_metric)
        avg_auc_value = np.mean(list(auc_value_dict.values()))

        if mode.lower() == 'test':
            # Save outputs to csv
            if save_outputs:
                mode_list = ['test'] * len(patient_ids)
                misc.save_predictions(patient_ids=patient_ids, endpoint_list=endpoint_list,
                                      y_pred_list_dict=y_pred_list_dict, y_true_list_dict=y_true_list_dict,
                                      mode_list=mode_list, model_name=model_name,
                                      exp_dir=exp_dir, logger=logger)
            return loss_value, auc_value_dict, patient_ids, y_pred_list_dict, y_true_list_dict

        # Prints
        logger.my_print(f'{mode} loss: {loss_value:.3f}.')
        logger.my_print(f'{mode} average AUC: {avg_auc_value:.3f}.')
        logger.my_print(f'{mode} AUCs...:')
        for endpoint in endpoint_list:
            logger.my_print(f'>>> {endpoint}: {auc_value_dict[endpoint]:.3f}.')

        return loss_value, auc_value_dict, avg_auc_value


def train(model, train_dataloader, val_dataloader, mean, std, optimizer, logger):
    """
    Train the model on the combined dataset, generate images, and save the images and the models.

    Args:
        model: PyTorch model
        train_dataloader: training dataloader
        val_dataloader: internal validation dataloader
        mean: normalization: list of means, one value for each channel (first value for CT, second value for RTDOSE)
        std: normalization: list of stds, one value for each channel (first value for CT, second value for RTDOSE)
        optimizer:
        logger: logger

    Returns:
        None
    """
    # Initiate variables
    best_val_loss_value = np.inf
    #best_val_avg_auc_value = -1
    best_val_avg_auc_value_dict = {endpoint: -1 for endpoint in endpoint_list}
    endpoint_unfrozen_list = copy.deepcopy(endpoint_list)  # list of endpoints for which the layers are not frozen yet
    endpoint_frozen_list = list()  # list of endpoints for which the layers are frozen
    train_avg_auc_values_list = list() if max_epochs > 0 else [None]
    val_avg_auc_values_list = list() if max_epochs > 0 else [None]
    train_loss_values_list = list() if max_epochs > 0 else [None]
    train_auc_values_list_dict = {endpoint: list() if max_epochs > 0 else [None] for endpoint in endpoint_list}
    val_loss_values_list = list() if max_epochs > 0 else [None]
    val_auc_values_list_dict = {endpoint: list() if max_epochs > 0 else [None] for endpoint in endpoint_list}
    lr_list = list() if max_epochs > 0 else [None]
    batch_size_list = list() if max_epochs > 0 else [None]
    nr_epochs_not_improved_dict = {endpoint: 0 for endpoint in endpoint_list}  # (EarlyStopping)

    train_num_iterations = len(train_dataloader)
    logger.my_print('Number of training iterations per epoch: {}.'.format(train_num_iterations))
    epoch = 0
    best_epoch_dict = {endpoint: 0 for endpoint in endpoint_list}
    freeze_epoch_dict = {endpoint: None for endpoint in endpoint_list}
    for epoch in range(max_epochs):
        # Print training stats
        logger.my_print('-' * 20)
        logger.my_print(f'Epoch {epoch + 1}/{max_epochs}...')
        for param_group in optimizer.param_groups:
            logger.my_print('Learning rate: {}.'.format(param_group['lr']))
            lr_list.append(param_group['lr'])
        cur_batch_size = train_dataloader.batch_size
        logger.my_print('Batch size: {}.'.format(cur_batch_size))
        batch_size_list.append(cur_batch_size)

        # Initiate training
        model.train()
        train_loss_value = 0
        train_y_pred_dict = dict()
        train_y_true_dict = dict()
        for endpoint in endpoint_list:
            train_y_pred_dict[endpoint] = torch.as_tensor([], dtype=torch.float32, device=device)
            train_y_true_dict[endpoint] = torch.as_tensor([], dtype=torch.int8, device=device)

        for i, batch_data in tqdm(enumerate(train_dataloader)):
            # Load data
            train_inputs, train_features, train_label_list = (
                batch_data['ct_dose_seg'].to(device),
                batch_data['features'].to(device),
                batch_data['label_list'].to(device)
            )
            train_label_dict = dict()
            # Note: swap train_label_list (tensor) from shape (batch_size, len(endpoint_list)) to
            # (len(endpoint_list), batch_size), so that each for-loop iteration contains the endpoint's values
            for i, (endpoint, train_label) in enumerate(zip(endpoint_list, torch.swapaxes(train_label_list, 0, 1))):
                train_label_dict[endpoint] = train_label

            if config.perform_augmix:
                for b in range(len(train_inputs)):
                    # Generate a random 32-bit integer
                    seed_b = random.getrandbits(32)
                    train_inputs[b] = process_data.aug_mix(arr=train_inputs[b], mixture_width=config.mixture_width,
                                                           mixture_depth=config.mixture_depth, augmix_strength=config.augmix_strength,
                                                           device=device, seed=seed_b)

            # Preprocess inputs and features
            # NOTE: these operations should also be performed in lr_finder.py
            train_inputs = process_data.preprocess_inputs(inputs=train_inputs, ct_mean=mean[0], ct_std=std[0],
                                                          rtdose_mean=mean[1], rtdose_std=std[1])
            train_features = process_data.preprocess_features(features=train_features)

            # Zero the parameter gradients and make predictions
            optimizer.zero_grad(set_to_none=True)
            train_outputs_dict = model(x=train_inputs, features=train_features)

            # Calculate loss
            train_loss = 0
            train_nr_of_endpoints_with_loss = 0
            for endpoint in endpoint_list:
                # Only consider patients with non-missing value for `endpoint`, i.e., train_labels in `valid_endpoint_values`
                train_batch_size_endpoint_before = train_outputs_dict[endpoint].shape[0]
                train_outputs_dict[endpoint] = train_outputs_dict[endpoint][
                    torch.isin(train_label_dict[endpoint], torch.as_tensor(valid_endpoint_values, device=device))]
                train_label_dict[endpoint] = train_label_dict[endpoint][
                    torch.isin(train_label_dict[endpoint], torch.as_tensor(valid_endpoint_values, device=device))]

                train_outputs = train_outputs_dict[endpoint]
                train_labels = train_label_dict[endpoint]
                train_batch_size_endpoint_after = train_outputs_dict[endpoint].shape[0]

                if train_batch_size_endpoint_after > 0:
                    try:
                        # Cross-Entropy, Ranking, Custom
                        train_loss += loss_function(train_outputs, train_labels)  # * (train_batch_size_endpoint_after / train_batch_size_endpoint_before)
                    except:
                        # BCE
                        train_loss += loss_function(train_outputs,
                                                    torch.reshape(train_labels, train_outputs.shape).to(train_outputs.dtype))  # * (train_batch_size_endpoint_after / train_batch_size_endpoint_before)
                    train_nr_of_endpoints_with_loss += 1

            # Scale loss value: has impact on weight updates! Else, the weights will be updated
            # `train_nr_of_endpoints_with_loss` times more aggresively than normal
            train_loss /= train_nr_of_endpoints_with_loss

            if train_loss.grad_fn:
              if optimizer_name in ['ada_hessian']:
                  # https://github.com/pytorch/pytorch/issues/4661
                  # https://discuss.pytorch.org/t/how-to-backward-the-derivative/17662?u=bpetruzzo
                  # Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient
                  # which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this.
                  # However, if we do use backward(create_graph=True), then we have to make sure to reset the
                  # .grad fields of our parameters to None after use to break the cycle and avoid the leak.
                  train_loss.backward(create_graph=True)
                  # torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
              else:
                  train_loss.backward()
            else:
              print("no grad")

            # Perform gradient clipping
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            if grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_max_norm)

            # Update model weights
            optimizer.step()

            # Reset the .grad fields of our parameters to None after use to break the cycle and avoid the memory leak
            if optimizer_name in ['ada_hessian']:
                for p in model.parameters():
                    p.grad = None

            # Scheduler: step() called after every batch update
            # Note: some schedulers generally update the LR every epoch instead of every batch, but epoch-wise update
            # will mess up LearningRateWarmUp. Batch-wise LR update is always valid and works with LearningRateWarmUp.
            if scheduler_name in ['cosine', 'exponential']:  # , 'step']:
                scheduler.step(epoch + (i + 1) / train_num_iterations)
            elif scheduler_name in ['cyclic']:
                scheduler.step()

            # Evaluate model (training set)
            train_loss_value += train_loss.item()

            for endpoint in endpoint_list:
                train_y_pred_dict[endpoint] = torch.cat([train_y_pred_dict[endpoint], train_outputs_dict[endpoint]], dim=0)
                train_y_true_dict[endpoint] = torch.cat([train_y_true_dict[endpoint], train_label_dict[endpoint]], dim=0)

        # Averaging training loss
        train_loss_value /= train_num_iterations

        # Compute training AUC
        train_y_pred_list_dict = dict()
        train_y_true_list_dict = dict()
        train_auc_value_dict = dict()
        for endpoint in endpoint_list:
            if loss_function_name in ['bce']:
                train_y_pred_list_dict[endpoint] = misc.proba(sigmoid_act(train_y_pred_dict[endpoint]))
            else:
                train_y_pred_list_dict[endpoint] = [softmax_act(i) for i in train_y_pred_dict[endpoint]]
            train_y_true_list_dict[endpoint] = [to_onehot(i) for i in train_y_true_dict[endpoint]]

            train_auc_value_dict[endpoint] = misc.compute_auc(y_pred_list=train_y_pred_list_dict[endpoint],
                                                              y_true_list=train_y_true_list_dict[endpoint],
                                                              auc_metric=auc_metric)
        train_avg_auc_value = np.mean(list(train_auc_value_dict.values()))
        train_avg_auc_values_list.append(train_avg_auc_value)

        # Prints
        logger.my_print(f'Training loss: {train_loss_value:.3f}.')
        logger.my_print(f'Training average AUC: {train_avg_auc_value:.3f}.')
        logger.my_print('Training AUCs...:')
        for endpoint in endpoint_list:
            logger.my_print(f'>>> {endpoint}: {train_auc_value_dict[endpoint]:.3f}.')

        train_loss_values_list.append(train_loss_value)
        for endpoint in endpoint_list:
            train_auc_values_list_dict[endpoint] += [train_auc_value_dict[endpoint]]

        # Perform plotting, internal validation, independent testing, and potentially freezing layers
        if (epoch + 1) % config.eval_interval == 0:
            # Plot DL model input of random patients
            if (epoch + 1 == 1) or ((epoch + 1) % config.plot_interval == 0):
                misc.plot_model_inputs(train_inputs, mean, std, epoch)

            # Perform internal validation
            val_loss_value, val_auc_value_dict, val_avg_auc_value = validate(model=model, dataloader=val_dataloader,
                                                                             mean=mean, std=std,
                                                                             mode='internal validation', logger=logger,
                                                                             save_outputs=True)
            val_loss_values_list.append(val_loss_value)
            val_avg_auc_values_list.append(val_avg_auc_value)

            for endpoint in endpoint_list:
                val_auc_values_list_dict[endpoint].append(val_auc_value_dict[endpoint])

            # Check if model performance improved for each endpoint
            endpoint_unfrozen_improved_dict = dict()
            endpoint_unfrozen_tolerance_dict = dict()
            for endpoint in endpoint_unfrozen_list:
                endpoint_unfrozen_improved_dict[endpoint] = (val_auc_value_dict[endpoint] > best_val_avg_auc_value_dict[endpoint])
                if endpoint_unfrozen_improved_dict[endpoint]:
                    nr_epochs_not_improved_dict[endpoint] = 0
                else:
                    nr_epochs_not_improved_dict[endpoint] += 1

                endpoint_unfrozen_tolerance_dict[endpoint] = (val_auc_value_dict[endpoint] > (max(val_auc_values_list_dict[endpoint]) - auc_tolerance))
            
            logger.my_print('Did the performance of unfrozen endpoints improve? (True/False): {}'.format(endpoint_unfrozen_improved_dict))
            logger.my_print('Number of consecutive epochs not improved: {}'.format(nr_epochs_not_improved_dict))

            # Save model if all endpoints improved w.r.t. current best values, or if at least one endpoint improved
            # w.r.t. current best values while the other unfrozen endpoints still perform better than best
            # values - tolerance
            if (sum(endpoint_unfrozen_improved_dict.values()) == len(endpoint_unfrozen_list)) or (
                    (sum(endpoint_unfrozen_improved_dict.values()) > 0) and
                    (sum(endpoint_unfrozen_tolerance_dict.values()) == len(endpoint_unfrozen_list))):

                if sum(endpoint_unfrozen_improved_dict.values()) == len(endpoint_unfrozen_list):
                    logger.my_print('Model performance for all endpoints improved.')
                else:
                    logger.my_print('Model performance for {} out of {} unfrozen endpoints improved, while '
                                    'the performance for all endpoints is still better than max(AUC) - tolerance'.
                        format(sum(endpoint_unfrozen_improved_dict.values()), len(endpoint_unfrozen_list)))
                best_val_avg_auc_value_dict = val_auc_value_dict
                best_val_loss_value = val_loss_value
                best_val_avg_auc_value = val_avg_auc_value
                for endpoint in endpoint_unfrozen_list:
                    best_epoch_dict[endpoint] = epoch + 1
                    # If the following is enabled, then consider performances better than "max(AUC) - tolerance" good
                    # enough and so do not add counter to EarlyStopping. This makes sense, e.g.: dysphagia keeps
                    # improving while xerostomia stays between AUC = 0.74 - 0.75 for the remaining epochs, then it
                    # can be helpful to not freeze the shared (+ xerostomia) layers yet. Note: shared + xerostomia layer
                    # will still be frozen if xerostomia performance dropped below max(AUC) - tolerance for
                    # `patience` consecutive epochs, however the counter will be reset if xerostomia performance
                    # is better than max(AUC) - tolerance within `patience` epochs, but that is fine because the model
                    # is performing well
                    if endpoint_unfrozen_tolerance_dict[endpoint]:
                        nr_epochs_not_improved_dict[endpoint] = 0

                torch.save(model.state_dict(), os.path.join(config.exp_dir, config.filename_best_model_pth))
                torch.save(optimizer.state_dict(), os.path.join(config.exp_dir, config.filename_best_optimizer_pth))
                torch.save(scheduler.state_dict(), os.path.join(config.exp_dir, config.filename_best_scheduler_pth))
                logger.my_print('Saved new best model.')

            logger.my_print('Best internal validation AUCs...:')
            for endpoint in endpoint_list:
                logger.my_print(f'>>> {endpoint}: {best_val_avg_auc_value_dict[endpoint]:.3f}.')
            logger.my_print(
                f'Corresponding average internal validation AUC: {best_val_avg_auc_value:.3f} at epoch: {best_epoch_dict[endpoint]}.')
            logger.my_print(
                f'Corresponding internal validation loss: {best_val_loss_value:.3f} at epoch: {best_epoch_dict[endpoint]}.')

        # EarlyStopping
        # If all endpoints did not improve within `patience` epochs, then stop training
        if sum([x >= patience for x in nr_epochs_not_improved_dict.values()]) == len(endpoint_list):
            logger.my_print('No internal validation improvement during the last {} consecutive epochs for all endpoints. '
                            'Stop training.'.format(nr_epochs_not_improved_dict[endpoint]))
            for endpoint in copy.deepcopy(endpoint_unfrozen_list):
                freeze_epoch_dict[endpoint] = epoch + 1
                endpoint_unfrozen_list.remove(endpoint)
                endpoint_frozen_list.append(endpoint)
            assert (len(endpoint_unfrozen_list) == 0) and (len(endpoint_frozen_list) == len(endpoint_list))
            return (train_loss_values_list, val_loss_values_list, 
                    train_avg_auc_values_list, val_avg_auc_values_list,
                    train_auc_values_list_dict, val_auc_values_list_dict,
                    lr_list, batch_size_list, epoch + 1, best_epoch_dict, freeze_epoch_dict)

        # If one or more (but not all) unfrozen endpoints did not improve within `patience` epochs, then freeze
        #   the shared layers and the layers of those non-improved endpoints
        elif max([nr_epochs_not_improved_dict[endpoint] for endpoint in endpoint_unfrozen_list]) >= patience:
            # Determine endpoints for which the layers should be frozen
            for endpoint in copy.deepcopy(endpoint_unfrozen_list):
                if nr_epochs_not_improved_dict[endpoint] >= patience:
                    logger.my_print('({}) No internal validation improvement during the last {} consecutive epochs. '
                                    'Freeze layers.'.format(endpoint, nr_epochs_not_improved_dict[endpoint]))
                    freeze_epoch_dict[endpoint] = epoch + 1
                    endpoint_unfrozen_list.remove(endpoint)
                    endpoint_frozen_list.append(endpoint)

            # Freeze shared and endpoint layers
            # Load best model so far. Note: we have to continue with the model from `patience` epochs ago!
            model.load_state_dict(torch.load(os.path.join(exp_dir, config.filename_best_model_pth)))
            optimizer.load_state_dict(torch.load(os.path.join(exp_dir, config.filename_best_optimizer_pth)))
            scheduler.load_state_dict(torch.load(os.path.join(exp_dir, config.filename_best_scheduler_pth)))

            for name, param in model.named_parameters():
                # if param.requires_grad and ('_shared_' or '{}'.format(endpoint)) in name:
                if param.requires_grad and any(x for x in ['_shared_'] + endpoint_frozen_list if x in name):
                    param.requires_grad = False
                logger.my_print('Layer name: {}; param.requires_grad: {}'.format(name, param.requires_grad))
            logger.my_print('')

            # Let us be nice to our model: reset nr_epochs_not_improved_dict[endpoint] to 0, so the model has again
            # `patience` epochs to improve
            for endpoint in endpoint_unfrozen_list:
                nr_epochs_not_improved_dict[endpoint] = 0

    return (train_loss_values_list, val_loss_values_list,
            train_avg_auc_values_list, val_avg_auc_values_list,
            train_auc_values_list_dict, val_auc_values_list_dict,
            lr_list, batch_size_list, epoch + 1, best_epoch_dict, freeze_epoch_dict)




if __name__ == '__main__':
    # Make sure that data_preproc_config.use_umcg == True
    use_umcg = data_preproc_config.use_umcg
    assert use_umcg

    # Initialize variables
    torch_version = config.torch_version
    seed = config.seed
    device = config.device
    exp_dir = config.exp_dir
    cv_folds = config.cv_folds
    # Deep Learning model
    dataloader_drop_last = config.dataloader_drop_last
    valid_endpoint_values = config.valid_endpoint_values
    model_name = config.model_name
    endpoint_list = config.endpoint_list
    label_smoothing = config.label_smoothing
    num_classes = config.num_classes
    pretrained_path = config.pretrained_path
    weight_init_name = config.weight_init_name
    kaiming_a = config.kaiming_a
    kaiming_mode = config.kaiming_mode
    kaiming_nonlinearity = config.kaiming_nonlinearity
    gain = config.gain
    optimizer_name = config.optimizer_name
    loss_function_name = config.loss_function_name
    lr_finder_num_iter = config.lr_finder_num_iter
    scheduler_name = config.scheduler_name
    grad_max_norm = config.grad_max_norm
    nr_of_decimals = config.nr_of_decimals
    # max_epochs = config.max_epochs
    batch_size = config.batch_size
    patience = config.patience
    auc_tolerance = config.auc_tolerance
    # Logistic regression model
    patient_id_col = data_preproc_config.patient_id_col
    patient_id_length = data_preproc_config.patient_id_length
    submodels_features_list = config.submodels_features_list
    features_lr_list = config.features_lr_list
    lr_coefficients_list = config.lr_coefficients_list

    # Checks
    assert auc_tolerance >= 0

    # Set seed for reproducibility
    torch.manual_seed(seed=seed)
    set_determinism(seed=seed)
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # Initialize classes
    sigmoid_act = torch.nn.Sigmoid()
    softmax_act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=config.num_ohe_classes)
    auc_metric = ROCAUCMetric()

    # Initialize data objects
    train_auc_list_dict = {endpoint: list() for endpoint in endpoint_list}
    train_auc_lr_list_dict = {endpoint: list() for endpoint in endpoint_list}
    val_auc_list_dict = {endpoint: list() for endpoint in endpoint_list}
    val_auc_lr_list_dict = {endpoint: list() for endpoint in endpoint_list}
    train_auc_value_dict = {endpoint: None for endpoint in endpoint_list}
    val_auc_value_dict = {endpoint: None for endpoint in endpoint_list}
    test_auc_value_dict = {endpoint: None for endpoint in endpoint_list}
    train_auc_list_mean_dict = {endpoint: None for endpoint in endpoint_list}
    val_auc_list_mean_dict = {endpoint: None for endpoint in endpoint_list}
    test_y_pred_sum_dict = {endpoint: None for endpoint in endpoint_list}
    test_y_sum_dict = {endpoint: None for endpoint in endpoint_list}
    test_y_pred_lr_sum_dict = {endpoint: None for endpoint in endpoint_list}
    test_y_lr_sum_dict = {endpoint: None for endpoint in endpoint_list}
    norm_mean_dict, norm_std_dict = dict(), dict()
    if model_name in ['efficientnet-b{}'.format(i) for i in range(9)] + \
            ['efficientnetv2_{}'.format(i) for i in ['s', 'm']] + \
            ['resnet_original' + 'resnext_original']:
        # Only batch_size > 1 is allowed for batch_norm
        assert batch_size > 1
        dataloader_drop_last = True

    for fold in range(cv_folds):
        # Clear console prints
        os.system('cls||clear')

        # Initialize variables
        logger = initialize(torch_version=torch_version, device=device)
        logger.my_print('Model name: {}'.format(model_name))
        logger.my_print('Fold: {}'.format(fold))
        logger.my_print('Seed: {}'.format(seed))
        test_loss_value, test_auc_value, test_auc_mean, test_auc_lr, test_auc_lr_mean = [None] * 5
        if pretrained_path is not None:
            pretrained_path_i = os.path.join(pretrained_path, str(fold), config.filename_best_model_pth)
        else:
            pretrained_path_i = None
        start = time.time()

        # Cross-Validation
        if cv_folds > 1:
            if fold == 0:
                # Fetch training, internal validation and test files
                cv_dict_0, cv_dict_1, test_dict = load_data.get_files(logger=logger)
                # Combine training and internal validation files into one set (which we can do cross-val on)
                cv_dict = cv_dict_0 + cv_dict_1

                # Perform Stratified CV
                if config.cv_type == 'stratified':
                    logger.my_print('Performing stratified {}-fold CV.'.format(cv_folds))
                    # Create stratified CV class

                    # DANIEL: added roundablout way to do strat sampling with multi-label data
                    cv_object = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)  
                    # Determine endpoint values to stratify on
                    cv_y = [x['label_list'] for x in cv_dict]
                    cv_y = [''.join([str(i) for i in l]) for l in cv_y]  # turn each multi-label list into a sting (e.g. [0-1110-1] for [0, -1, 1, 1, 0, -1])
                    cv_y = [np.where(np.array(list(dict.fromkeys(cv_y)))==e)[0][0]for e in cv_y]  # encode those into an integer

                    # For each fold: determine train and val indices
                    cv_idx_list = list()
                    for idx in cv_object.split(X=cv_dict, y=cv_y):
                        cv_idx_list.append(idx)

            # For this fold: select training and internal validation indices
            train_idx, valid_idx = cv_idx_list[fold]

            # Fetch training set using cv indices in 'cv_idx_list'
            train_dict = list()
            for i in train_idx:
                train_dict.append(cv_dict[i])

            # Fetch internal validation set using cv indices in 'cv_idx_list'
            val_dict = list()
            for i in valid_idx:
                val_dict.append(cv_dict[i])
            assert len(train_dict) + len(val_dict) == len(cv_dict)

        # Training-validation-test split
        else:
            train_dict, val_dict, test_dict = [None] * 3

        (train_dataloader, val_dataloader, test_dataloader, train_ds, dl_class, train_dataloader_args_dict, batch_size, channels,
         depth, height, width, n_train_0_list, n_train_1_list, n_val_0_list, n_val_1_list, n_test_0_list, n_test_1_list,
         n_features, train_dict, val_dict, test_dict, norm_mean, norm_std) = \
            load_data.main(train_dict=train_dict, val_dict=val_dict, test_dict=test_dict,
                           fold=fold, drop_last=dataloader_drop_last, logger=logger)
        
        logger.my_print('Input shape: {}.'.format([batch_size, channels, depth, height, width]))
        num_batches_per_epoch = len(train_dataloader)
        norm_mean_dict[fold] = norm_mean
        norm_std_dict[fold] = norm_std

        # Initialize model
        model = misc.get_model(channels=channels, depth=depth, height=height, width=width, n_features=n_features,
                               pretrained_path=pretrained_path_i, logger=logger)
        model.to(device=device)

        # Weight initialization
        if 'selu' in model_name:
            # Source: https://github.com/bioinf-jku/SNNs/tree/master/Pytorch
            weight_init_name = 'kaiming_normal'
            kaiming_a = None
            kaiming_mode = 'fan_in'
            kaiming_nonlinearity = 'linear'

        logger.my_print('Weight init name: {}.'.format(weight_init_name))
        if pretrained_path_i is not None:
            logger.my_print('Using pretrained weights: {}.'.format(pretrained_path_i))
        elif ((pretrained_path_i is None) and
              (weight_init_name is not None) and
              ('efficientnet' not in model_name) and
              ('convnext' not in model_name) and
              ('resnext' not in model_name)):
            # Source: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
            model.apply(lambda m: misc.weights_init(m=m, weight_init_name=weight_init_name, kaiming_a=kaiming_a,
                                                    kaiming_mode=kaiming_mode,
                                                    kaiming_nonlinearity=kaiming_nonlinearity, gain=gain,
                                                    logger=logger))
            logger.my_print('Using our own weights init scheme for {}.'.format(model_name))
        else:
            logger.my_print('Using default PyTorch weights init scheme for {}.'.format(model_name))

        # Create and save model summary
        # summary() only accepts non-zero integers, i.e. n_features = 0 is invalid in `input_size`.
        # However n_features = 0 in get_model() will make `n_features` in summary() redundant, so
        # we just use `1` in the input of summary().
        # Important note: torchinfo.summary() somehow uses some random() on input_size, because it will create
        # random input. I.e. input_size changes the next random values.
        total_params = misc.get_model_summary(model=model, input_size=[(batch_size, channels, depth, height, width),
                                                                       (batch_size, max(n_features, 1))], device=device,
                                              logger=logger)
        # Compile model (PyTorch 2)  #os.name = 'nt' is windows, torch.compile is not supported on windows
        if torch_version.startswith('2.') and os.name != 'nt' and config.device == torch.device('cuda'):
            #model = torch.compile(model, mode='default', backend='inductor')
            num_GPUs = torch.cuda.device_count()
            logger.my_print(f'Model compiled! Running on {num_GPUs} GPUs')
        else:
            logger.my_print('Windows OR no GPU = No model compiling.')


        # Initial model
        logger.my_print('Initial model:')
        for name, param in model.named_parameters():
            logger.my_print('Layer name: {}; param.requires_grad: {}'.format(name, param.requires_grad))
        logger.my_print('')

        # Initialize loss function
        loss_function = misc.get_loss_function()

        # Initialize optimizer
        lr = config.lr
        optimizer = misc.get_optimizer(optimizer_name=optimizer_name, model=model, lr=lr, 
                                       num_batches_per_epoch=num_batches_per_epoch, logger=logger)

        # Perform learning rate finder
        if (config.lr_finder_num_iter > 0) and (max_epochs > 0):
            lr, optimizer = misc.learning_rate_finder(
                model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer,
                optimizer_name=optimizer_name, loss_function=loss_function, mean=norm_mean_dict[fold],
                std=norm_std_dict[fold], logger=logger)
        starting_lr = lr

        # If lr = None (i.e. lr cannot be found), then skip iteration of the for-loop
        if lr is None:
            # continue
            max_epochs = 0
        else:
            max_epochs = config.max_epochs

        # Initialize scheduler
        # Note: should be performed after learning_rate_finder()
        scheduler = misc.get_scheduler(scheduler_name=scheduler_name, optimizer=optimizer, optimal_lr=lr, 
                                       num_batches_per_epoch=num_batches_per_epoch, logger=logger)

        # Train model
        (train_loss_values, val_loss_values,
         train_avg_auc_values_list, val_avg_auc_values_list,
         train_auc_values_dict, val_auc_values_dict,
         lr_values, batch_size_values, epochs, best_epoch_dict, freeze_epoch_dict) = \
            train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                  mean=norm_mean_dict[fold], std=norm_std_dict[fold], optimizer=optimizer, logger=logger)

        if max_epochs > 0:
            # Plot training and internal validation losses
            y_list = [[train_loss_values, val_loss_values]] +\
                      [[train_avg_auc_values_list, val_avg_auc_values_list]] + \
                      [[train_auc_values_dict[endpoint], val_auc_values_dict[endpoint]] for endpoint in endpoint_list] + \
                      [[lr_values], [batch_size_values]]
            y_label_list = ['Loss', 'Average AUC'] + ['{} AUC'.format(x) for x in endpoint_list] + ['LR', 'Batch_size']
            best_epoch_list = [None, max(enumerate(val_avg_auc_values_list), key=lambda x: x[1])[0] + 1] + \
                              [best_epoch_dict[endpoint] for endpoint in endpoint_list] + [None, None]
            freeze_epoch_list = [None, None] + [freeze_epoch_dict[endpoint] for endpoint in endpoint_list] + [None, None]
            legend_list = [['Training', 'Internal validation']] * (len(y_list) - 2) + [None, None]
            misc.plot_values(y_list=y_list, y_label_list=y_label_list, best_epoch_list=best_epoch_list,
                             freeze_epoch_list=freeze_epoch_list, legend_list=legend_list,
                             figsize=config.figsize, save_filename=os.path.join(exp_dir, config.filename_results_png))

            # Load best model
            model.load_state_dict(torch.load(os.path.join(exp_dir, config.filename_best_model_pth)))

            # Store predictions on training and validation set
            # IMPORTANT NOTE: train_dataloader still performs data augmentation!
            train_loss_value, train_auc_value_dict, train_patient_ids, train_y_pred_list_dict, train_y_list_dict = \
                validate(model=model, dataloader=train_dataloader, mean=norm_mean_dict[fold], std=norm_std_dict[fold],
                         mode='test', logger=logger, save_outputs=False)
            val_loss_value, val_auc_value_dict, val_patient_ids, val_y_pred_list_dict, val_y_list_dict = validate(
                model=model, dataloader=val_dataloader, mean=norm_mean_dict[fold], std=norm_std_dict[fold], mode='test',
                logger=logger, save_outputs=False)
            all_patient_ids = train_patient_ids + val_patient_ids
            all_modes = ['train'] * len(train_patient_ids) + ['val'] * len(val_patient_ids)

            all_y_pred_list_dict = dict()
            all_y_list_dict = dict()
            for endpoint in endpoint_list:
                all_y_pred_list_dict[endpoint] = train_y_pred_list_dict[endpoint] + val_y_pred_list_dict[endpoint]
                all_y_list_dict[endpoint] = train_y_list_dict[endpoint] + val_y_list_dict[endpoint]

                train_auc_list_dict[endpoint].append(train_auc_value_dict[endpoint])
                val_auc_list_dict[endpoint].append(val_auc_value_dict[endpoint])

            # Test evaluate model
            if test_dataloader is not None:
                test_loss_value, test_auc_value_dict, test_patient_ids, test_y_pred_list_dict, test_y_list_dict = \
                    validate(model=model, dataloader=test_dataloader, mean=norm_mean_dict[fold], std=norm_std_dict[fold],
                             mode='test', logger=logger, save_outputs=True)

                # Store predictions on test set
                all_patient_ids += test_patient_ids
                all_modes += ['test'] * len(test_patient_ids)

                for endpoint in endpoint_list:
                    all_y_pred_list_dict[endpoint] += test_y_pred_list_dict[endpoint]
                    all_y_list_dict[endpoint] += test_y_list_dict[endpoint]

                    if (test_y_pred_sum_dict[endpoint] is None) and (test_y_sum_dict[endpoint] is None):
                        test_y_pred_sum_dict[endpoint] = test_y_pred_list_dict[endpoint]
                        test_y_sum_dict[endpoint] = test_y_list_dict[endpoint]
                    else:
                        test_y_pred_sum_dict[endpoint] = [x + y for x, y in zip(test_y_pred_sum_dict[endpoint], test_y_pred_list_dict[endpoint])]
                        test_y_sum_dict[endpoint] = [x + y for x, y in zip(test_y_sum_dict[endpoint], test_y_list_dict[endpoint])]

            # Save predictions on training, validation and test set
            misc.save_predictions(patient_ids=all_patient_ids, endpoint_list=endpoint_list,
                                  y_pred_list_dict=all_y_pred_list_dict, y_true_list_dict=all_y_list_dict,
                                  mode_list=all_modes, model_name=model_name + '_all',
                                  exp_dir=exp_dir, logger=logger)

        # Run logistic regression (LR) model (reference model)
        features_csv = os.path.join(config.data_dir, config.filename_stratified_sampling_test_csv)
        patient_ids_json = os.path.join(exp_dir, config.filename_train_val_test_patient_ids_json.format(fold))
        (train_patient_ids_lr, train_y_pred_lr_dict, train_y_lr_dict,
         val_patient_ids_lr, val_y_pred_lr_dict, val_y_lr_dict,
         test_patient_ids_lr, test_y_pred_lr_dict, test_y_lr_dict, lr_coefficients) = \
            run_logistic_regression(features_csv=features_csv, patient_id_col=patient_id_col,
                                    endpoint_list=endpoint_list, valid_endpoint_values=valid_endpoint_values,
                                    submodels_features_list=submodels_features_list,
                                    features_list=features_lr_list,
                                    lr_coefficients_list=lr_coefficients_list,
                                    patient_id_length=patient_id_length, patient_ids_json=patient_ids_json, seed=seed,
                                    nr_of_decimals=nr_of_decimals, logger=logger)

        # Compute training and internal validation AUC of LR model
        train_y_pred_lr_list_dict, train_y_lr_list_dict, train_auc_lr_dict = dict(), dict(), dict()
        val_y_pred_lr_list_dict, val_y_lr_list_dict, val_auc_lr_dict = dict(), dict(), dict()

        for endpoint in endpoint_list:
            train_y_pred_lr_list_dict[endpoint] = [x for x in train_y_pred_lr_dict[endpoint]]
            train_y_lr_list_dict[endpoint] = [to_onehot(i) for i in train_y_lr_dict[endpoint]]
            train_auc_lr_dict[endpoint] = misc.compute_auc(y_pred_list=train_y_pred_lr_list_dict[endpoint],
                                                           y_true_list=train_y_lr_list_dict[endpoint],
                                                           auc_metric=auc_metric)
            train_auc_lr_list_dict[endpoint].append(train_auc_lr_dict[endpoint])

            val_y_pred_lr_list_dict[endpoint] = [x for x in val_y_pred_lr_dict[endpoint]]
            val_y_lr_list_dict[endpoint] = [to_onehot(i) for i in val_y_lr_dict[endpoint]]
            val_auc_lr_dict[endpoint] = misc.compute_auc(y_pred_list=val_y_pred_lr_list_dict[endpoint],
                                                         y_true_list=val_y_lr_list_dict[endpoint],
                                                         auc_metric=auc_metric)
            val_auc_lr_list_dict[endpoint].append(val_auc_lr_dict[endpoint])

        # Store predictions on training and validation set of LR model
        # IMPORTANT NOTE: train_dataloader still performs data augmentation!
        all_patient_ids_lr = train_patient_ids_lr + val_patient_ids_lr
        all_modes_lr = ['train'] * len(train_patient_ids_lr) + ['val'] * len(val_patient_ids_lr)

        all_y_pred_lr_list_dict = dict()
        all_y_lr_list_dict = dict()
        for endpoint in endpoint_list:
            all_y_pred_lr_list_dict[endpoint] = train_y_pred_lr_list_dict[endpoint] + val_y_pred_lr_list_dict[endpoint]
            all_y_lr_list_dict[endpoint] = train_y_lr_list_dict[endpoint] + val_y_lr_list_dict[endpoint]

        # Test evaluate (LR model)
        if test_dataloader is not None:

            # Store predictions on test set
            all_patient_ids_lr += test_patient_ids_lr
            mode_lr_list = ['test'] * len(test_patient_ids_lr)
            all_modes_lr += mode_lr_list

            test_y_pred_lr_list_dict = dict()
            test_y_lr_list_dict = dict()
            test_auc_lr_dict = dict()
            for endpoint in endpoint_list:
                test_y_pred_lr_list_dict[endpoint] = [x for x in test_y_pred_lr_dict[endpoint]]
                test_y_lr_list_dict[endpoint] = [to_onehot(i) for i in test_y_lr_dict[endpoint]]
                test_auc_lr_dict[endpoint] = misc.compute_auc(y_pred_list=test_y_pred_lr_list_dict[endpoint],
                                                              y_true_list=test_y_lr_list_dict[endpoint],
                                                              auc_metric=auc_metric)

                all_y_pred_lr_list_dict[endpoint] += test_y_pred_lr_list_dict[endpoint]
                all_y_lr_list_dict[endpoint] += test_y_lr_list_dict[endpoint]

            # Save outputs to csv
            misc.save_predictions(patient_ids=test_patient_ids_lr, endpoint_list=endpoint_list,
                                  y_pred_list_dict=test_y_pred_lr_list_dict, y_true_list_dict=test_y_lr_list_dict,
                                  mode_list=mode_lr_list, model_name='lr', exp_dir=exp_dir, logger=logger)

            for endpoint in endpoint_list:
                if (test_y_pred_lr_sum_dict[endpoint] is None) and (test_y_lr_sum_dict[endpoint] is None):
                    test_y_pred_lr_sum_dict[endpoint] = test_y_pred_lr_list_dict[endpoint]
                    test_y_lr_sum_dict[endpoint] = test_y_lr_list_dict[endpoint]
                else:
                    test_y_pred_lr_dict[endpoint] = [x for x in test_y_pred_lr_list_dict[endpoint]]
                    test_y_pred_lr_sum_dict[endpoint] = [x + y for x, y in zip(test_y_pred_lr_sum_dict[endpoint], test_y_pred_lr_list_dict[endpoint])]
                    test_y_lr_sum_dict[endpoint] = [x + y for x, y in zip(test_y_lr_sum_dict[endpoint], test_y_lr_list_dict[endpoint])]

            # Ensemble (DL and LR model)
            if (fold == cv_folds - 1) and (cv_folds > 1):
                # Make sure that test_y_sum and test_y_lr_sum have only `num_classes` different tensors:
                # e.g. cv_folds = 5: {(0.0, 5.0), (5.0, 0.0)}
                if max_epochs > 0:
                    test_y_pred_ens_dict = dict()
                    test_y_ens_dict = dict()
                    for endpoint in endpoint_list:
                        assert len(set([tuple(x.tolist()) for x in test_y_sum_dict[endpoint]])) == num_classes
                        test_y_pred_ens_dict[endpoint] = [x / cv_folds for x in test_y_pred_sum_dict[endpoint]]
                        test_y_ens_dict[endpoint] = [x / cv_folds for x in test_y_sum_dict[endpoint]]
                    misc.save_predictions(patient_ids=test_patient_ids, endpoint_list=endpoint_list,
                                          y_pred_list_dict=test_y_pred_ens_dict, y_true_list_dict=test_y_ens_dict,
                                          mode_list=mode_lr_list, model_name=model_name + '_ens', exp_dir=exp_dir, logger=logger)

                test_y_pred_lr_ens_dict = dict()
                test_y_lr_ens_dict = dict()
                for endpoint in endpoint_list:
                    assert len(set([tuple(x.tolist()) for x in test_y_lr_sum_dict[endpoint]])) == num_classes
                    test_y_pred_lr_ens_dict[endpoint] = [x / cv_folds for x in test_y_pred_lr_sum_dict[endpoint]]
                    test_y_lr_ens_dict[endpoint] = [x / cv_folds for x in test_y_lr_sum_dict[endpoint]]

                misc.save_predictions(patient_ids=test_patient_ids_lr, endpoint_list=endpoint_list,
                                      y_pred_list_dict=test_y_pred_lr_ens_dict, y_true_list_dict=test_y_lr_ens_dict,
                                      mode_list=mode_lr_list, model_name='lr_ens', exp_dir=exp_dir, logger=logger)

        # Save predictions on training, validation and test set
        misc.save_predictions(patient_ids=all_patient_ids_lr, endpoint_list=endpoint_list,
                              y_pred_list_dict=all_y_pred_lr_list_dict, y_true_list_dict=all_y_lr_list_dict,
                              mode_list=all_modes_lr, model_name='lr_all', exp_dir=exp_dir, logger=logger)

        if max_epochs > 0:
            # Create results.csv
            misc.create_results(train_loss=train_loss_value, val_loss=val_loss_value, test_loss=test_loss_value,
                                train_auc_dict=train_auc_value_dict, train_auc_lr_dict=train_auc_lr_dict, val_auc_dict=val_auc_value_dict,
                                val_auc_lr_dict=val_auc_lr_dict, test_auc_dict=test_auc_value_dict, test_auc_lr_dict=test_auc_lr_dict,
                                seed=seed, epochs=epochs, best_epoch_dict=best_epoch_dict, batch_size=batch_size,
                                starting_lr=starting_lr, patience=patience, model_name=model_name,
                                features_dl=config.features_dl, n_features=n_features, filters=config.filters,
                                kernel_sizes=config.kernel_sizes, strides=config.strides, lrelu_alpha=config.lrelu_alpha,
                                pooling_conv_filters=config.pooling_conv_filters, linear_units=config.linear_units,
                                dropout_p=config.dropout_p, use_bias=config.use_bias, total_params=total_params,
                                loss_weights=config.loss_weights,
                                n=[[x + y for x, y in zip(n_train_0_list, n_train_1_list)],
                                   [x + y for x, y in zip(n_val_0_list, n_val_1_list)],
                                   [x + y for x, y in zip(n_test_0_list, n_test_1_list)]],
                                n_train=[x + y for x, y in zip(n_train_0_list, n_train_1_list)],
                                n_val=[x + y for x, y in zip(n_val_0_list, n_val_1_list)],
                                n_test=[x + y for x, y in zip(n_test_0_list, n_test_1_list)],
                                train_val_test_frac=[config.train_frac, config.val_frac,
                                                     1 - config.train_frac - config.val_frac],
                                cv_folds=cv_folds, input_shape=[batch_size, channels, depth, height, width],
                                normalization_mean_dict=norm_mean_dict, normalization_std_dict=norm_std_dict,
                                ct_b_min_max=[config.ct_b_min, config.ct_b_max],
                                rtdose_b_min_max=[config.rtdose_b_min, config.rtdose_b_max],
                                seg_orig_labels=config.seg_orig_labels, seg_target_labels=config.seg_target_labels,
                                interpolation_mode_3d=[config.ct_interpol_mode_3d, config.rtdose_interpol_mode_3d,
                                                       config.segmentation_interpol_mode_3d],
                                interpolation_mode_2d=[config.ct_interpol_mode_2d, config.rtdose_interpol_mode_2d,
                                                       config.segmentation_interpol_mode_2d],
                                perform_augmix=config.perform_augmix, weight_init_name=weight_init_name,
                                optimizer=optimizer_name, loss_function=loss_function_name, label_weights=config.label_weights,
                                label_smoothing=label_smoothing, lr_finder_num_iter=lr_finder_num_iter,
                                scheduler=scheduler_name, grad_max_norm=grad_max_norm, lr_coefficients=lr_coefficients)


        end = time.time()
        logger.my_print('Elapsed time: {time} (H:M:S).'.format(time=time.strftime('%H:%M:%S', time.gmtime(end - start))))
        logger.my_print('DONE!')
        logger.close()

        # Rename folder
        src_folder_name = os.path.join(exp_dir)
        dst_folder_name = os.path.join(exp_dir +
                                       '_{}'.format(int(fold)) +
                                       '_{}'.format(seed) +
                                       '_{}'.format(fold) +
                                       '_{}'.format(max(best_epoch_dict.values())) +
                                       '_{}'.format(model_name) +
                                       '_params_{}'.format(total_params) +
                                       '_auc' +
                                       '_tr_{}'.format(round(np.mean([x for x in train_auc_value_dict.values() if x is not None]), nr_of_decimals)) +
                                       '_lr_{}'.format(round(np.mean([x for x in train_auc_lr_dict.values() if x is not None]), nr_of_decimals)) +
                                       '_val_{}'.format(round(np.mean([x for x in val_auc_value_dict.values() if x is not None]), nr_of_decimals)) +
                                       '_lr_{}'.format(round(np.mean([x for x in val_auc_lr_dict.values() if x is not None]), nr_of_decimals))
                                       )

        if test_dataloader is not None:
            dst_folder_name += '_test_{}'.format(round(np.mean([x for x in test_auc_value_dict.values() if x is not None]), nr_of_decimals))
            dst_folder_name += '_lr_{}'.format(round(np.mean([x for x in test_auc_lr_dict.values() if x is not None]), nr_of_decimals))

        # Results of all folds (avg and ensemble)
        if (fold == cv_folds - 1) and (cv_folds > 1):
            dst_folder_name += '_avg'

            if max_epochs > 0:
                # DL training (avg)
                train_auc_list_mean_dict = dict()
                for endpoint in endpoint_list:
                    train_auc_list_dict[endpoint] = [x for x in train_auc_list_dict[endpoint] if x is not None]
                    train_auc_list_mean_dict[endpoint] = round(
                        sum(train_auc_list_dict[endpoint]) / len(train_auc_list_dict[endpoint]), nr_of_decimals)

            # LR training (avg)
            train_auc_lr_list_mean_dict = dict()
            for endpoint in endpoint_list:
                train_auc_lr_list_dict[endpoint] = [x for x in train_auc_lr_list_dict[endpoint] if x is not None]
                train_auc_lr_list_mean_dict[endpoint] = round(
                    sum(train_auc_lr_list_dict[endpoint]) / len(train_auc_lr_list_dict[endpoint]), nr_of_decimals)

            dst_folder_name += '_tr_{}_lr_{}'.format(round(np.mean([x for x in train_auc_list_mean_dict.values() if x is not None]), nr_of_decimals),
                                                     round(np.mean([x for x in train_auc_lr_list_mean_dict.values() if x is not None]), nr_of_decimals))

            # Internal validation (avg) and test (ensemble)
            # DL model
            dst_folder_name += '_val'
            val_auc_list_mean_dict = dict()
            test_auc_mean_dict = dict()
            if max_epochs > 0:
                # Validation (avg) and test (ensemble)
                for endpoint in endpoint_list:
                    val_auc_list_dict[endpoint] = [x for x in val_auc_list_dict[endpoint] if x is not None]
                    val_auc_list_mean_dict[endpoint] = round(
                        sum(val_auc_list_dict[endpoint]) / len(val_auc_list_dict[endpoint]), nr_of_decimals)

                    # Test (ensemble)
                    test_auc_mean_dict[endpoint] = misc.compute_auc(y_pred_list=test_y_pred_ens_dict[endpoint],
                                                                    y_true_list=test_y_ens_dict[endpoint],
                                                                    auc_metric=auc_metric)

            dst_folder_name += '_{}'.format(round(np.mean([x for x in val_auc_list_mean_dict.values() if x is not None]), nr_of_decimals))

            # LR model
            # Validation (avg)
            val_auc_lr_list_mean_dict = dict()
            for endpoint in endpoint_list:
                val_auc_lr_list_dict[endpoint] = [x for x in val_auc_lr_list_dict[endpoint] if x is not None]
                val_auc_lr_list_mean_dict[endpoint] = round(
                    sum(val_auc_lr_list_dict[endpoint]) / len(val_auc_lr_list_dict[endpoint]), nr_of_decimals)

            dst_folder_name += '_lr_{}'.format(round(np.mean([x for x in val_auc_lr_list_mean_dict.values() if x is not None]), nr_of_decimals))

            # Test (ensemble)
            test_auc_lr_mean_dict = dict()
            for endpoint in endpoint_list:
                test_auc_lr_mean_dict[endpoint] = misc.compute_auc(y_pred_list=test_y_pred_lr_ens_dict[endpoint],
                                                                   y_true_list=test_y_lr_ens_dict[endpoint],
                                                                   auc_metric=auc_metric)
            dst_folder_name += '_ens_{}'.format(round(np.mean([x for x in test_auc_mean_dict.values() if x is not None]), nr_of_decimals))
            dst_folder_name += '_lr_{}'.format(round(np.mean([x for x in test_auc_lr_mean_dict.values() if x is not None]), nr_of_decimals))

        shutil.move(src_folder_name, dst_folder_name)
