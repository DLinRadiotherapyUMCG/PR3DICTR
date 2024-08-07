import numpy as np
import torch
import os
from data_preproc.data_preproc_functions import create_folder_if_not_exists
from torch import nn

def freeze_model_layers(model, layers_to_freeze):
    """
    Freezes the layers of a model.
    Args:
        model: The model to freeze.
        layers_to_freeze: A list of layers to freeze. Any model modules with a name containing any of these strings will be frozen.
    Returns:
        model: The model, but with the specified layers frozen.
    """
    print("FREEZING: ", layers_to_freeze)
    for name, module in model.named_parameters():
        if any((elem in name) for elem in layers_to_freeze):
            module.requires_grad = False
            #print("    frozen", name)
    return model


def save_model_output_head_weights(config, model, output_heads_to_save):
    """
    Saves the output heads of a model to a temporary folder.
    Args:
        config: The config object.
        model: The model to save.
        output_heads_to_save: A list of output heads to save (i.e. the names of the endpoints)
    Returns:
        None
    """
    create_folder_if_not_exists(config.temp_model_weights_dir)

    print("SAVING: ", output_heads_to_save)
    for name, module in model.named_modules():
        decomposed_name = name.split('.')
        # if the last element of the module is the endpoint, save it, as it contains all of the layers within the output head
        # this is better than just "if decomposed_name in output_heads_to_save" because it doesn't save each layer separately
        if decomposed_name[-1] in output_heads_to_save:
            # delete the old file if it exists
            save_dir = os.path.join(config.temp_model_weights_dir, name+".pth")
            # if os.path.exists(save_dir):
            #     os.remove(save_dir)
            #print("   saving -- ", name)
            # save the weights of this output head to the temporary folder
            torch.save(module.state_dict(), save_dir)

def load_model_output_head_weights(config, model, output_heads_to_load, freeze_loaded_heads=True):
    """
    Loads the weights of specific output heads of a model from a temporary folder. These weights should correspond to the "best weights" for this part of the model.

    Args:
        config: The config object.
        model: The model to load the output heads into.
        output_heads_to_load: A list of output heads to load (i.e. the names of the endpoints)
        freeze_laoded_heads: Whether or not to freeze the loaded heads (default = `True`).
    Returns:
        model: The model, but with the specified output heads replaced.
    """
    print("LOADING: ", output_heads_to_load)

    for name, module in model.named_modules():
        decomposed_name = name.split('.')
        
        # if ((elem in output_heads_to_load) for elem in decomposed_name):    # this would save each layer separately
        if decomposed_name[-1] in output_heads_to_load:
            #print("  loading ---", name)
            # overwrite the loaded weights onto the current model
            loaded_weights_dict = torch.load(os.path.join(config.temp_model_weights_dir, name+".pth"))
            module.load_state_dict(loaded_weights_dict)
            del loaded_weights_dict

            # freeze the weights of this output head (if not done so already)
            if freeze_loaded_heads:
                module.requires_grad_ = False
                for param in module.parameters():
                    param.requires_grad = False

    return model


def freeze_normalisation_layers(model):
    """
    Freezes the Batch and Instance normalisation layers of a pytorch model.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            module.track_running_stats = False
            for param in module.parameters():
                param.requires_grad = False
                param.track_running_stats = False




def adjust_optimiser(cfg, model, optimizer):
    """
    Resets the optimizer to only update the parameters that require gradients.
    This is needed if we have frozen some layers of the model.
    Args:
        cfg: The config object.
        model: The model.
        optimizer: The optimizer.
    Returns:
        optimizer: The optimizer, which now only considers weights that require training (i.e. that are not frozen).
    
    """
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    optimizer.param_groups[0]['params'] = params_to_update

    return optimizer




def layer_freezing(cfg, model, optimizer, scheduler, best_metrics_tracker, model_backbone_frozen,
                   epoch,
                    val_loss_dict, val_loss_value,
                    val_avg_auc_value, val_auc_value_dict, 
                    logger):
    
    
    best_val_auc_dict = best_metrics_tracker.best_val_auc_dict
    best_val_avg_auc = best_metrics_tracker.best_val_avg_auc
    best_val_avg_loss = best_metrics_tracker.best_val_avg_loss
    best_val_loss_dict = best_metrics_tracker.best_val_loss_dict
    best_val_epoch_num = best_metrics_tracker.best_val_epoch_num
    
    
    """
    Function that manages the early stopping by freezing the model's layers.
    """
    STOP_TRAINING = False

    nr_epochs_not_improved_dict = best_metrics_tracker.nr_epochs_not_improved_dict
    nr_epochs_not_improved = best_metrics_tracker.nr_epochs_not_improved

    endpoint_list = cfg.endpoint_list
    endpoint_unfrozen_list = list(nr_epochs_not_improved_dict.keys())
    endpoint_frozen_list = [x for x in cfg.endpoint_list if x not in endpoint_unfrozen_list]

    if not model_backbone_frozen and cfg.use_mean_loss_early_stopping:
        improved_this_epoch, nr_epochs_not_improved = check_mean_loss_improved(cfg, val_loss_value, best_val_avg_loss, nr_epochs_not_improved)
        logger.my_print(f"Mean loss improved: {nr_epochs_not_improved}")
    else:    
        improved_this_epoch, nr_epochs_not_improved_dict = check_endpoints_loss_improved(cfg, val_loss_dict, best_val_loss_dict, nr_epochs_not_improved_dict)
        logger.my_print(f"Endpoints loss improved: {nr_epochs_not_improved_dict}")

    # get names of endpoints which have not improved for too many epochs
    endpoints_surpassed_patience = [key for (key, value) in nr_epochs_not_improved_dict.items() if value >= cfg.patience]
    endpoints_improved = [key for (key, value) in nr_epochs_not_improved_dict.items() if value == 0]


    # if the model backbone is not frozen
    if model_backbone_frozen == False:

        # if all endpoints improved
        
        if (cfg.use_mean_loss_early_stopping==False and all(value == 0 for value in nr_epochs_not_improved_dict.values())) or \
            (cfg.use_mean_loss_early_stopping==True and nr_epochs_not_improved == 0):
            
            assert len(endpoints_surpassed_patience) == 0 and len(endpoint_frozen_list) == 0

            logger.my_print(" ALL ENDPOINTS IMPROVED")
            #nr_epochs_not_improved = 0
            best_val_epoch_num = epoch
            best_val_avg_loss = val_loss_value
            best_val_avg_auc = val_avg_auc_value
            best_val_auc_dict = val_auc_value_dict
            best_val_loss_dict = val_loss_dict
            
            # Save the model
            torch.save(model.state_dict(), os.path.join(cfg.exp_dir, cfg.filename_best_model_pth))
            torch.save(optimizer.state_dict(), os.path.join(cfg.exp_dir, cfg.filename_best_optimizer_pth))
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(cfg.exp_dir, cfg.filename_best_scheduler_pth))
            
            # also save all of the output heads, just in case
            save_model_output_head_weights(config=cfg, model=model, output_heads_to_save=endpoints_improved)   

            # set the new best epoch to the logger
            logger.set_best_epoch_dict(best_auc_dict=val_auc_value_dict, loss=val_loss_value, 
                                    avg_auc=val_avg_auc_value, epoch_n=best_val_epoch_num)

        # if (at least) one of them has not improved for too many epochs, freeze the output head and the model backbone
        if (cfg.use_mean_loss_early_stopping==False and len(endpoints_surpassed_patience) > 0) or \
            (cfg.use_mean_loss_early_stopping==True and nr_epochs_not_improved > cfg.patience):
        #if len(endpoints_surpassed_patience) > 0:
            # freeze shared layers and the relevant non-shared endpoint layers
            logger.my_print('FREEZING BACKBONE')
            model_backbone_frozen = True
            
            # reload the 'best model', and freeze the backbone and endpoints in `endpoints_surpassed_patience`
            model.load_state_dict(torch.load(os.path.join(cfg.exp_dir, cfg.filename_best_model_pth)))
            optimizer.load_state_dict(torch.load(os.path.join(cfg.exp_dir, cfg.filename_best_optimizer_pth)))
            if scheduler is not None:
                scheduler.load_state_dict(torch.load(os.path.join(cfg.exp_dir, cfg.filename_best_scheduler_pth)))

            model = load_model_output_head_weights(config=cfg, model=model, 
                                                    output_heads_to_load=endpoint_list, freeze_loaded_heads=False)
            
            layers_to_freeze =  ["_shared_", "shared_"] # + endpoints_surpassed_patience # freeze the shared layers. "_shared_" must be in the name of these shared layers
            model = freeze_model_layers(model=model, layers_to_freeze=layers_to_freeze)   # NOTE: put the whole endpoint list here to freeze everything

            optimizer = adjust_optimiser(cfg, model, optimizer)

            for endpoint in endpoints_surpassed_patience:
                nr_epochs_not_improved_dict[endpoint] = 0

        logger.print_epoch_table(mode='epoch')

    # if the model backbone is frozen (and some endpoints may be frozen)
    else:
        # if any endpoints improved
        if len(endpoints_improved) > 0:  
            for endpoint in endpoints_improved:
                if best_val_loss_dict[endpoint] > val_loss_dict[endpoint]:
                    best_val_loss_dict[endpoint] = val_loss_dict[endpoint]
                #if val_auc_value_dict[endpoint] > best_val_auc_dict[endpoint]:
                    best_val_auc_dict[endpoint] = val_auc_value_dict[endpoint]

            # if an endpoint improved, save the output head
            save_model_output_head_weights(config=cfg, model=model, output_heads_to_save=endpoints_improved)

            best_val_epoch_num = epoch
            best_val_avg_loss = val_loss_value
            #print(" BEST ", best_val_auc_dict)
            best_val_avg_auc = np.mean([best_val_auc_dict[endpoint] for endpoint in endpoint_list])

            logger.set_best_epoch_dict(best_auc_dict=best_val_auc_dict, loss=best_val_avg_loss, 
                                        avg_auc=best_val_avg_auc, epoch_n=best_val_epoch_num)

        # if (at least) one of them has not improved for too many epochs
        if len(endpoints_surpassed_patience) > 0:
            logger.my_print('FREEZING ENDPOINTS:', endpoints_surpassed_patience)
            # reload the best output head, and freeze it. This function will also freeze this head
            model = load_model_output_head_weights(config=cfg, model=model, 
                                                    output_heads_to_load=endpoints_surpassed_patience, freeze_loaded_heads=True)

            model = freeze_model_layers(model=model, layers_to_freeze=endpoints_surpassed_patience)

            optimizer = adjust_optimiser(cfg, model, optimizer)

            # remove the endpoint from the unfrozen dict
            for endpoint in endpoints_surpassed_patience:
                endpoint_unfrozen_list.remove(endpoint)
                nr_epochs_not_improved_dict.pop(endpoint)

                endpoint_frozen_list.append(endpoint)

        logger.print_epoch_table(mode='epoch')

        # if all endpoints are frozen, stop training
        if len(endpoint_unfrozen_list) == 0:
            logger.my_print(f'EARLY STOPPING AT EPOCH: {epoch}.')
            STOP_TRAINING = True

    # make sure that the list of frozen and unfrozen endpoints is equal to the toal number of endpoints
    assert (len(endpoint_unfrozen_list) + len(endpoint_frozen_list)) == len(endpoint_list)

    best_metrics_tracker.update(nr_epochs_not_improved_dict = nr_epochs_not_improved_dict, 
                                nr_epochs_not_improved = nr_epochs_not_improved,
                                
                                best_val_auc_dict = best_val_auc_dict,
                                best_val_avg_auc = best_val_avg_auc,
                                best_val_avg_loss = best_val_avg_loss,
                                best_val_loss_dict = best_val_loss_dict,
                                best_val_epoch_num = best_val_epoch_num
    )

    
    return STOP_TRAINING, model, optimizer, scheduler, logger, best_metrics_tracker, model_backbone_frozen





def check_mean_loss_improved(cfg, val_loss_value, best_val_avg_loss, nr_epochs_not_improved):
    improved_this_epoch = True
    if round(val_loss_value, cfg.nr_of_decimals) < round(best_val_avg_loss, cfg.nr_of_decimals):
        nr_epochs_not_improved = 0
    else:
        nr_epochs_not_improved += 1
        improved_this_epoch = False

    return improved_this_epoch, nr_epochs_not_improved


def check_endpoints_loss_improved(cfg, val_loss_dict, best_val_loss_dict, nr_epochs_not_improved_dict):
    improved_this_epoch = True
    for endpoint in list(nr_epochs_not_improved_dict.keys()):
        if round(val_loss_dict[endpoint], cfg.nr_of_decimals) < round(best_val_loss_dict[endpoint], cfg.nr_of_decimals) + cfg.loss_tolerance:
            nr_epochs_not_improved_dict[endpoint] = 0
        else:
            nr_epochs_not_improved_dict[endpoint] += 1
            improved_this_epoch = False
    
    return improved_this_epoch, nr_epochs_not_improved_dict



def early_stopping(cfg, model, optimizer, scheduler,  best_metrics_tracker, logger,
                   epoch, val_loss_value, val_loss_dict, 
                    val_avg_auc_value, val_auc_value_dict):

    STOP_TRAINING = False

    best_val_avg_loss = best_metrics_tracker.best_val_avg_loss
    best_val_loss_dict = best_metrics_tracker.best_val_loss_dict
    nr_epochs_not_improved = best_metrics_tracker.nr_epochs_not_improved
    nr_epochs_not_improved_dict = best_metrics_tracker.nr_epochs_not_improved_dict

    # if using the mean loss for early stopping
    if cfg.use_mean_loss_early_stopping == True:
        improved_this_epoch, nr_epochs_not_improved = check_mean_loss_improved(cfg, val_loss_value, best_val_avg_loss, nr_epochs_not_improved)
        logger.my_print(f'Number of consecutive epochs not improved: {nr_epochs_not_improved}')
        # if the mean loss has surpassed the patience, stop training
        if nr_epochs_not_improved >= cfg.patience:
            STOP_TRAINING = True
    # if using the per-endpoint losses for early stopping
    else:
        improved_this_epoch, nr_epochs_not_improved_dict = check_endpoints_loss_improved(cfg, val_loss_dict, best_val_loss_dict, nr_epochs_not_improved_dict)
        logger.my_print(f'Number of consecutive epochs not improved, per endpoint: {nr_epochs_not_improved_dict}')
        # if any endpoint has surpassed the patience, stop training
        if any(value >= cfg.patience for value in nr_epochs_not_improved_dict.values()):
            STOP_TRAINING = True

    # save the patience counters
    best_metrics_tracker.update(nr_epochs_not_improved = nr_epochs_not_improved,
                                nr_epochs_not_improved_dict = nr_epochs_not_improved_dict) 

    if improved_this_epoch:
        best_val_epoch_num = epoch
        best_val_avg_loss = val_loss_value
        best_val_avg_auc_value = val_avg_auc_value
        best_val_auc_dict = val_auc_value_dict

        # if the loss improved, save the metrics
        best_metrics_tracker.update(best_val_auc_dict = best_val_auc_dict,
                                    best_val_avg_auc_value = best_val_avg_auc_value,
                                    best_val_avg_loss = best_val_avg_loss,
                                    best_val_loss_dict = val_loss_dict,
                                    best_val_epoch_num = best_val_epoch_num
    )   

        # set the new best epoch to the logger
        logger.set_best_epoch_dict(best_auc_dict=val_auc_value_dict, loss=val_loss_value, 
                                    avg_auc=val_avg_auc_value, epoch_n=best_val_epoch_num)

        # Save the model
        torch.save(model.state_dict(), os.path.join(cfg.exp_dir, cfg.filename_best_model_pth))
        torch.save(optimizer.state_dict(), os.path.join(cfg.exp_dir, cfg.filename_best_optimizer_pth))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(cfg.exp_dir, cfg.filename_best_scheduler_pth))
        
    
    # print the stats of the epoch with the best validation results
    logger.print_epoch_table(mode='epoch')
    
    #logger.my_print(f'Number of consecutive epochs not improved: {nr_epochs_not_improved_dict}')


    return STOP_TRAINING, model, optimizer, scheduler, logger, best_metrics_tracker

