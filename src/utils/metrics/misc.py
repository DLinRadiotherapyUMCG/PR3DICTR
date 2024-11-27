"""
Miscellaneous functions.
"""
import sys
import os
import math
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    AsDiscrete,
)

from src.visualization.plot_slices import plot_slices


def create_results(config, **kwargs):
    """
    Create results.csv. 
    Args:
        config: the config object (for the output dir and filenames)
        **kwargs: as many key-value pairs (variable names and respective values) as you want to save to the csv

    Returns:

    """
    df = pd.DataFrame([x for x in kwargs.values()],
                      index=[x for x in kwargs.keys()],
                      columns=[config.exp_name])
    df.to_csv(os.path.join(config.exp_dir, config.filename_results_csv), sep=';')


def append_to_results_file(config, **kwargs):
    """
    Loads results.csv, and appends new rows with extra results (e.g. the test or external validation results)
    Args:
        config: the config object (for the output dir and filenames)
        **kwargs: as many key-value pairs (variable names and respective values) as you want to save to the csv

    Returns:

    """
    # load the old results.csv
    df_old = pd.read_csv(os.path.join(config.exp_dir, config.filename_results_csv), sep=';', index_col=0)
    # make a small dataframe with the new results
    df_new = pd.DataFrame([x for x in kwargs.values()],
                      index=[x for x in kwargs.keys()],
                      columns=[config.exp_name])
    # append the new results to the old dataframe
    df = pd.concat([df_old, df_new])
    # save the new dataframe to the results.csv (overwrites the old results.csv)
    df.to_csv(os.path.join(config.exp_dir, config.filename_results_csv), sep=';')



def create_ensemble_results(config, **kwargs):
    """
    Create results.csv. 
    Args:
        config: the config object (for the output dir and filenames)
        **kwargs: as many key-value pairs (variable names and respective values) as you want to save to the csv

    Returns:

    """
    df = pd.DataFrame([x for x in kwargs.values()],
                      index=[x for x in kwargs.keys()],
                      columns=[config.exp_name])
    df.to_csv(os.path.join(config.exp_dir, config.filename_results_overview_csv), sep=';')


def append_to_ensemble_results_file(config, **kwargs):
    """
    Loads results.csv, and appends new rows with extra results (e.g. the test or external validation results)
    Args:
        config: the config object (for the output dir and filenames)
        **kwargs: as many key-value pairs (variable names and respective values) as you want to save to the csv

    Returns:
    """
    # load the old results.csv
    df_old = pd.read_csv(os.path.join(config.exp_dir, config.filename_results_overview_csv), sep=';', index_col=0)
    # make a small dataframe with the new results
    df_new = pd.DataFrame([x for x in kwargs.values()],
                      index=[x for x in kwargs.keys()],
                      columns=[config.exp_name])
    # append the new results to the old dataframe
    df = pd.concat([df_old, df_new])
    # save the new dataframe to the results.csv (overwrites the old results.csv)
    df.to_csv(os.path.join(config.exp_dir, config.filename_results_overview_csv), sep=';')

    print(os.path.join(config.exp_dir, config.filename_results_overview_csv))





def get_all_df_stats(config, df_split, logger):
    total_size = len(df_split)
    split_col = config.split_col
    df_train = df_split[df_split[split_col] == 'train']
    df_val = df_split[df_split[split_col] == 'val']
    df_test = df_split[df_split[split_col] == 'test']

    test_frac = config.test_frac
    if config.cv_folds == 1:
        train_frac = config.train_frac
        val_frac = config.val_frac
    else: 
        # in case of cross validation, the train_frac and val_frac are calculated based on the 
        # test_frac and number of folds (i.e. they are not == what is in the config)
        val_frac = (1-test_frac) / config.cv_folds
        train_frac = 1 - test_frac - val_frac
        
    get_df_stats(config=config, df=df_split, mode='full', frac=1, total_size=total_size, logger=logger)
    get_df_stats(config=config, df=df_train, mode='train', frac=train_frac, total_size=total_size, logger=logger)
    get_df_stats(config=config, df=df_val, mode='val', frac=val_frac, total_size=total_size, logger=logger)
    get_df_stats(config=config, df=df_test, mode='test', frac=test_frac, total_size=total_size, logger=logger)


def get_df_stats(config, df, mode, frac, total_size, logger, decimals=3):
    """
    Print dataframe statistics, useful for printing results of Stratified Sampling

    Args:
        df:
        groups:
        mode:
        frac:
        total_size:
        logger,
        decimals:

    Returns:

    """
    size = len(df)
    logger.my_print('>>> {} <<<'.format(mode.upper()))
    logger.my_print('Expected size (fraction): {} ({}%)'.format(int(total_size * frac), round(frac * 100, decimals)))
    logger.my_print('Size (fraction): {} ({}%)'.format(size, round(size / total_size * 100, decimals)))
    for g in config.strata_groups:
        logger.my_print('Group: {}'.format(g))

        # Non-NaNs
        g_values = [x for x in set(df[g]) if not pd.isna(x)]
        for v in g_values:
            g_count = sum(df[g] == v)
            g_perc = sum(df[g] == v) / size * 100
            logger.my_print('\tValue = {}, \tProportion = {}% ({})'.format(v, round(g_perc, decimals), g_count))

        # NaNs (if any)
        g_count = sum(df[g].isnull())
        if g_count > 0:
            g_perc = sum(df[g].isnull()) / size * 100
            logger.my_print('\tValue = {}, \tProportion = {}% ({})'.format(np.nan, round(g_perc, decimals), g_count))
        logger.my_print('')


def intersection(list_1, list_2):
    """
    Intersection of two lists.
    Args:
        list_1:
        list_2:
    Returns:

    """
    return [x for x in list_1 if x in list_2]




def plot_values(y_list, y_label_list, best_epoch_list, freeze_epoch_list, legend_list, figsize, save_filename):
    """
    Create and save line plot of a list of loss values (per epoch).

    Args:
        y_list (list): list of list of values to be plotted (e.g. list of loss values and list of AUC metric)
        y_label_list (list): list of label of y
        best_epoch_list (list): list of ints, draw vertical line at x=best_epoch
        freeze_epoch_list (list): list of ints, draw vertical line at x=freeze_epoch
        legend_list (list): list of legend names
        figsize (tuple): size of output figure
        save_filename (str): filename to save to, including path

    Returns:

    """
    fig, ax = plt.subplots(nrows=len(y_list), ncols=1, figsize=tuple(figsize))
    ax = ax.flatten()

    for i, (y, y_label, best_epoch, freeze_epoch, legend) in enumerate(
            zip(y_list, y_label_list, best_epoch_list, freeze_epoch_list, legend_list)):
        
        for y_i in y:
            epochs = [e + 1 for e in range(len(y_i))]
            ax[i].plot(epochs, y_i)
        ax[i].set_ylim(bottom=0)
        ax[i].set_xlabel('Epoch')
        if best_epoch is not None:
            ax[i].axvline(x=best_epoch, color='red', linestyle='--')
        if freeze_epoch is not None:
            ax[i].axvline(x=freeze_epoch, color='cyan', linestyle='--')
        if y_label is not None:
            ax[i].set_title(y_label)
        if legend is not None:
            ax[i].legend(legend, bbox_to_anchor=(1, 1), loc='upper left')
        # Add horizontal gridlines
        ax[i].grid(axis='y', linestyle='dotted')

    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close(fig)


def save_predictions(config, patient_ids, endpoint_list, y_pred_list_dict, y_true_list_dict, mode_list, model_name, exp_dir, logger, external_set=False):
    """
    Save prediction and corresponding true labels to csv.

    Args:
        patient_ids (list): list of patient ids
        endpoint_list (list): list of endpoints
        y_pred_list_dict (dict of lists): dict of lists of PyTorch tensors
        y_true_list_dict (dict of lists): dict of lists of PyTorch tensors
        mode_list (list): list of strings
        model_name (str):
        exp_dir (str):
        logger:

    Returns:

    """

    # Initialize df
    df_patient_ids = pd.DataFrame(patient_ids, columns=['PatientID'])
    df_mode = pd.DataFrame(mode_list, columns=['Mode'])
    df_y = pd.concat([df_patient_ids, df_mode], axis=1)

    for endpoint in endpoint_list:
        # Convert to CPU
        y_pred = [x.cpu().numpy() for x in y_pred_list_dict[endpoint]]
        y_true = [x.cpu().numpy() for x in y_true_list_dict[endpoint]]

        # Save to DataFrame
        if config.num_ohe_classes == 1:
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred'.format(endpoint)])
            df_y_true = pd.DataFrame(y_true, columns=['{}_true'.format(endpoint)])
        else:
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            num_cols = y_pred.shape[1]            
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred_{}'.format(endpoint, c) for c in range(num_cols)])
            df_y_true = pd.DataFrame(y_true, columns=['{}_true_{}'.format(endpoint, c) for c in range(num_cols)])
        df_y = pd.concat([df_y, df_y_pred, df_y_true], axis=1)

    # Save to file
    if external_set:
        #print(model_name)
        output_filename = os.path.join(exp_dir, config.filename_i_outputs_external_csv.format(i=model_name))
    else:
        output_filename = os.path.join(exp_dir, config.filename_i_outputs_csv.format(i=model_name))
    df_y.to_csv(output_filename, sep=';', index=False)




def plot_model_inputs(config, plot_inputs, epoch):
    """
    Plots the model inputs (CT, RTDOSE, Segmentation) used during training, and saves the figure as a png.
    Args:
        config (ConfigObject): configuration object
        plot_inputs (list): list of model input images
        epoch (int): current epoch number
    """
    for patient_idx in range(min(config.max_nr_images_per_interval, len(plot_inputs))):
        
        ct_range = (config.ct_a_max - config.ct_a_min)
        rtdose_range = (config.rtdose_a_max - config.rtdose_a_min)
        
        CT = (plot_inputs[patient_idx][0].cpu() * ct_range) + config.ct_a_min
        RTDOSE = plot_inputs[patient_idx][1].cpu() * rtdose_range
        RTSTRUCT = plot_inputs[patient_idx][2].cpu()

        plotting_rows_dicts = [
            {
                "Label": "CT",
                "CT": CT
            },
            {
                "Label": "RTDOSE",
                "RTDOSE": RTDOSE
            },
            {
                "Label": "RTSTRUCT",
                "RTSTRUCT": RTSTRUCT
            }
        ]

        slices = [20, 30, 40, 50, 60, 70, 80, 90]
        fig, axes = plot_slices(plotting_rows_dicts, slices)# , title=f"{patient_id} slices")

        filename=os.path.join(config.exp_figures_dir,'epoch_{}_idx_{}.png'.format(epoch, patient_idx))

        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)


####################################################################################################
        # New: Daniel









def apply_sigmoid_or_softmax(cfg, y_pred_dict, y_true_dict):
    """
    Applies sigmoid or softmax activation function to the model's predictions and one-hot encodes the true labels (if one-hot encoding is necessary).
    Args:
        y_pred_dict: A dictionary with the model's predictions for each endpoint, where each value is a tensor of predictions.
        y_true_dict: A dictionary with the true labels for each endpoint, in the same structure as `y_pred_dict`.
    Returns:
        y_pred_list_dict: A dictionary with the model's predictions for each endpoint, with softmax or sigmoid applied, 
                        and in the format that is used for computing AUCs and saving results (according to `cfg.ny_ohe_classes`).
        y_true_list_dict: A dictionary with the true labels for each endpoint, in the same structure as `y_pred_list_dict`.
    """
    sigmoid_act = torch.nn.Sigmoid()
    softmax_act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=cfg.num_ohe_classes)
    
    y_pred_list_dict = dict()
    y_true_list_dict = dict()

    for endpoint in cfg.endpoint_list:
        if cfg.loss_function_name in ['mse']: 
            # do not apply sigmoid or softmax to the predictions
            y_pred_list_dict[endpoint] = torch.squeeze(y_pred_dict[endpoint])
            y_true_list_dict[endpoint] = y_true_dict[endpoint]
        elif cfg.loss_function_name in ['bce'] or cfg.num_ohe_classes == 1:  # one value per toxicity endpoint
            # apply sigmoid function to the predictions
            y_pred_list_dict[endpoint] = torch.squeeze(sigmoid_act(y_pred_dict[endpoint]))
            y_true_list_dict[endpoint] = y_true_dict[endpoint]
        else:   # 2 values per toxicity endpoint [one hot encoded into 0 and 1 columns]
            # apply softmax function to the predictions
            y_pred_list_dict[endpoint] = [softmax_act(i) for i in y_pred_dict[endpoint]]
            y_true_list_dict[endpoint] = [to_onehot(i) for i in y_true_dict[endpoint]]

    return y_pred_list_dict, y_true_list_dict


    



def append_to_list_dicts(config, list_dicts = [], value_dicts = []):
    """
    
    """
    assert len(list_dicts) == len(value_dicts)

    for idx in range(len(list_dicts)):
        for endpoint in config.endpoint_list:
            list_dicts[idx][endpoint].append(value_dicts[idx][endpoint])
    
    return list_dicts


def summarise_results_over_folds(cfg, list_dict, test_set = False):
    """
    Takes in a dictionary of lists (one list per endpoint), returns a dictionary with the mean of each list (one mean per endpoint), 
    as well a total mean value (the mean of all endpoints)
    """
    
    mean_per_endpoint_dict = dict()

    for endpoint in cfg.endpoint_list:
        if test_set and not cfg.use_test_set:
            mean_per_endpoint_dict[endpoint] = None
        else:
            mean_per_endpoint_dict[endpoint] = np.mean(list_dict[endpoint])

    if test_set and not cfg.use_test_set:
        overall_mean = None
    else:
        overall_mean = np.mean([x for x in mean_per_endpoint_dict.values() if x is not None])

    return overall_mean, mean_per_endpoint_dict


def concatenate_predictions(config, train_y_pred_list_dict, train_y_list_dict, val_y_pred_list_dict, val_y_list_dict, test_y_pred_list_dict, test_y_list_dict):
    """
    
    Args:
        Six dicts of lists with the predictions and true labels for each train, val, test set.
    Returns:
        all_y_pred_list_dict: a dict of lists with all of the predictions for each endpoint
        all_y_true_list_dict: a dict of lists with all of the true labels for each endpoint
    """
    all_y_pred_list_dict = {endpoint: [] for endpoint in config.endpoint_list}
    all_y_true_list_dict = {endpoint: [] for endpoint in config.endpoint_list}

    sets = [[train_y_pred_list_dict, train_y_list_dict], [val_y_pred_list_dict, val_y_list_dict]]
    if config.use_test_set: 
        sets.append([test_y_pred_list_dict, test_y_list_dict])

    for [preds, labels] in sets:
        for endpoint in config.endpoint_list:
            all_y_pred_list_dict[endpoint] += preds[endpoint]
            all_y_true_list_dict[endpoint] += labels[endpoint]
       
    return all_y_pred_list_dict, all_y_true_list_dict



def sum_dictionaries(dict1, dict2, divisor = None):
    """
    Return the sum of two dictionaries that have identical keys.
    """
    
    if not divisor:
        sum_dict = {key: dict1[key] + dict2[key] for key in dict1}
    else:
        sum_dict = {key: dict1[key] + (dict2[key] / divisor) for key in dict1}
    
    return sum_dict

def append_to_list_dict(main_dict, append_dict):
    """
    Return the sum of two dictionaries that have identical keys.
    `main_dict` must be a dictionary with lists as values!
    Args:
        main_dict: A dictionary with lists as values
        append_dict: A dictionary with one value per key, to be appended to main_dict's lists
    Returns:
        main_dict with a value appended to each key's list
    """
    
    for key in main_dict:
        main_dict[key].append(append_dict[key])

    return main_dict


def set_random_seeds(cfg, set_new_random_seed=False):
    """
    Set the seed for reproducibility. Assumes the seed number set in the config file, unless  `random` is True, in which case a totally random seed is used.
    """
    if set_new_random_seed:  
        seed = np.random.randint(0, 1000) # set a new random seed for everything
        cfg.seed = seed
    else:
        seed = cfg.seed

    torch.manual_seed(seed=seed)
    if cfg.num_GPUs > 0:
        torch.cuda.manual_seed(seed=seed)
    set_determinism(seed=seed)
    random.seed(a=seed)
    np.random.seed(seed=seed)

    if cfg.reproducible:
        torch.backends.cudnn.deterministic = True

    return cfg


def generate_discrete_colormap(cmap_name='hsv', num_colors=20):
    N=num_colors
    arr = np.arange(N)/N
    N_up = int(math.ceil(N/7)*7)
    arr.resize(N_up)
    arr = arr.reshape(7, N_up//7).T.reshape(-1)
    ret = plt.cm.hsv(arr)
    n = ret[:,3].size
    a = n//2
    b = n-a
    for i in range(3):
        ret[0:n//2,i] *= np.arange(0.2,1,0.8/a)
    ret[n//2:, 3] *= np.arange(1,0.1,-0.9/b)
    return ret

def make_colorlist(config):
    colours = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:cyan']

    if len(config.endpoint_list) > len(colours):
        colours = generate_discrete_colormap(num_colors=len(config.endpoint_list))

    return colours

def plot_losses_dicts(config, train_losses_dict, val_losses_dict, mean_train_loss_list, mean_val_loss_list):
    """
    Makes a plot showing the training and validation losses for each endpoint per epoch, as well as the overall loss per epoch.
    """
    endpoint_names = deepcopy(list(train_losses_dict.keys()))
    colours = make_colorlist(config)


    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    fig, axes = plt.subplots()
    X = np.arange(1, len(mean_val_loss_list) + 1)

    for idx, endpoint_name in enumerate(endpoint_names):
        axes.plot(X, train_losses_dict[endpoint_name], color = colours[idx], linestyle='solid', linewidth=1)
        axes.plot(X, val_losses_dict[endpoint_name], color = colours[idx], linestyle='dashed', linewidth=1)

    axes.plot(X, mean_train_loss_list, label="Mean Train", color = "black", linestyle='solid', linewidth=1)
    axes.plot(X, mean_val_loss_list, label="Mean Val.", color = "black", linestyle='dashed', linewidth=1)

    # add dummies in order to get the legend to look nice
    if not config.loss_function_name == 'mse':
        colours.insert(0, 'black')
        endpoint_names.insert(0, '$\it{Average}$')
    #linestyles.insert(0, 'solid')
    dummy_color_lines = [axes.plot([], [], c=colours[idx], label=endpoint)[0] for idx, endpoint in enumerate(endpoint_names)]
    dummy_linestyles_lines = [axes.plot([], [], c='black', linestyle=linestyles[idx], label=set)[0] for idx, set in enumerate(["Train", "Validation"])]

    legend = fig.legend(handles=dummy_color_lines, title='$\\bf{Endpoint}$', 
                     bbox_to_anchor=(1, 0.95), loc='upper left')
    fig.add_artist(legend)

    linestyle_legend = fig.legend(handles=dummy_linestyles_lines, title='$\\bf{Dataset}$', 
                                bbox_to_anchor=(1.05, 0.5), loc='center left')
    fig.add_artist(linestyle_legend)

    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_title("Training and Validation Losses of Each Endpoint")
    fig.tight_layout()
    save_path = os.path.join(config.exp_dir, 'all_toxicities losses.png')
    fig.savefig(save_path,  bbox_inches='tight')
    plt.close(fig)




def set_torch_reproduciblity_backends(config):
    """
    Set the torch backends for reproducibility.
    Args:
        config: config object
    """
    if config.reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
