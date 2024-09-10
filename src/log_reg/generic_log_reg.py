# -*- coding: utf-8 -*-


import logging
import os

import pandas as pd
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from src.utils.log_table import log_table


def lr_load_data(config, model_index=None):
    """
    Load the training and validation data, this is different from the deep learning model as data is not converted to
    a PyTorch dataset. Images are neither loaded nor preprocessed.
    :param config:
    :param model_index:
    :return:
    """
    
    
    if model_index is None:
        train_data = pd.read_csv(os.path.join(config['paths']['csv'], 'train_full.csv'), delimiter=',',
                                 dtype={'PatientID': str})
    else:
        train_data = pd.read_csv(os.path.join(config['paths']['csv'], f'train_{model_index}.csv'), delimiter=',',
                                 dtype={'PatientID': str})
    val_data = pd.read_csv(os.path.join(config['paths']['csv'], 'valid.csv'), delimiter=',', dtype={'PatientID': str})
    return train_data, val_data


def lr_prepare_data(config, train_data, val_data):
    """
    Prepare the data for the logistic regression model:
    - Select the relevant columns
    - Extract the label
    :param config:
    :param train_data:
    :param val_data:
    :return:
    """
    relevant_columns = config['columns']['logistic_regression_features']
    label = config['log_reg']['label']
    X_train = train_data[relevant_columns]
    y_train = train_data[label]
    X_val = val_data[relevant_columns]
    y_val = val_data[label]
    return X_train, y_train, X_val, y_val


def lr_fit_model(config, X_train, y_train, X_val, y_val):
    """
    Fit the logistic regression model and calculate the AUC on the validation set
    :param config:
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :return:
    """
    # Fit the model
    model = LogisticRegression(random_state=config['seed'])
    
    # Messty things to handle any missing data
    X_train_cop = dict.fromkeys(X_train.keys())
    
    for key in X_train.keys():

        X_train_cop[key] = list(np.expand_dims(np.array(X_train[key]),axis = 1)[y_train!=-1])
        
    X_train = pd.DataFrame(X_train_cop)
    
    y_train = np.array(y_train)[y_train!=-1]
    
    # fit the model
    model.fit(X_train, y_train)
    
    # Calculate the AUC on the validation set
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_val_pred = np.expand_dims(y_val_pred, axis=1)
    y_val_pred = np.array(y_val_pred)[y_val!=-1]
    y_val = np.array(y_val)[y_val!=-1]
    auc = roc_auc_score(y_val, y_val_pred)
    return model, auc


def lr_print_model_coefficients(config, model):
    """
    Print the coefficients of the logistic regression model into a table
    :param config:
    :param model:
    :return:
    """
    relevant_columns = config['columns']['logistic_regression_features']
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature")
    table.add_column("Coefficient")
    for feature, coef in zip(relevant_columns, model.coef_[0]):
        table.add_row(feature, str(round(coef, 4)))
    table.add_row('intercept', str(model.intercept_[0]))
    logging.info(log_table(table))


def lr_save_model(config, model, model_index=None):
    """
    Save the logistic regression model to disk
    :param config:
    :param model:
    :param model_index: 
    :return:
    """
    model_path = os.path.join(config['paths']['output'],
                              f'lr_model{f"_{model_index}" if model_index is not None else ""}.pkl')
    logging.info(f'Saving model to {model_path}')
    pd.to_pickle(model, model_path)
    return
    
def train_lr(config):
    """
    """
    # Load the dataset
    train_data, val_data = lr_load_data(config)
    X_train, y_train, X_val, y_val = lr_prepare_data(config, train_data, val_data)

    # Initialize the model
    model, auc = lr_fit_model(config, X_train, y_train, X_val, y_val)   
    
    if config['log_reg']['print_cof']:
        lr_print_model_coefficients(config,model)
    if config['log_reg']['save_model']:
        lr_save_model(config,model)
        
    return model, auc 
















