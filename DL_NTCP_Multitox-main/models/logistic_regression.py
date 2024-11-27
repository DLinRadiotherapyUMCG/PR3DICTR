"""
Perform Logistic Regression on the same training-internal_validation-test split as Deep Learning model.

Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""
import json
import torch
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



@ignore_warnings(category=ConvergenceWarning)
def refit_logistic_regression(config, patient_id_col, logger):
    """
    Run logistic regression model on training (training + internal_validation) and test data.

    Note: if submodels_features and lr_coefficients are both None, then the logistic regression model will be
        directly fitted to features.

    Assumption: there are no overlapping of patient_ids between train, internal_validation and test set
        (see load_data.py).

    Args:
        config (Config): configuration object
        patient_id_col (str): name of the patient_id column
        submodels_features_list (list): list of lists of features for submodels
        features_list (list): list of features
        lr_coefficients_list (list): list of coefficients for submodels
        logger (Logger): logger object

    Returns:

    """   
    # submodels_features_list = config.submodels_features_list
    # features_lr_list = config.features_lr_list
    # lr_coefficients_list = config.lr_coefficients_list

    features_csv = os.path.join(config.data_dir, config.filename_stratified_sampling_test_csv)
    endpoint_list = config.endpoint_list
    patient_id_length = config.patient_id_length

    valid_endpoint_values = config.valid_endpoint_values
    seed = config.seed
    nr_of_decimals = config.nr_of_decimals

    patient_ids_json = os.path.join(config.exp_dir, config.filename_train_val_test_patient_ids_json)


    # Initialize variables
    train_y_pred_dict, train_y_dict = dict(), dict()
    val_y_pred_dict, val_y_dict = dict(), dict()
    test_y_pred_dict, test_y_dict = dict(), dict()

    # Load features + labels
    df = pd.read_csv(features_csv, sep=';', decimal='.')
    df[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df[patient_id_col]]

    learned_lr_coefficients = {}


    #df.loc[:, patient_id_col] = df.loc[:, patient_id_col].apply(lambda x: '%0.{}d'.format(patient_id_length) % int(x))

    for i, endpoint in enumerate(endpoint_list):
        logger.my_print('Fitting LR models for: {}.'.format(endpoint))
        
        endpoint_tox = endpoint.split("_")[0] # get only the name of the toxicty, not the endpoint time (e.g. only "Dysphagia", no "M06")
        # use that key to access the features lists. They are the same for all 'late' (post-treatment) CITOR models
        submodels_features = config.submodels_features_list_dict[endpoint_tox]
        features = config.features_lr_list_dict[endpoint_tox]
        lr_coefficients = config.lr_coefficients_list_dict[endpoint_tox]
        if lr_coefficients == "NONE": lr_coefficients = None

        # Check that we fit on the features directly (i.e. if submodels_features and lr_coefficients are both None),
        # fit submodels and construct final model from submodels (i.e. if submodels_features is not None) or
        # reuse coefficients (i.e. if lr_coefficients is not None)
        # if (submodels_features is None) and (lr_coefficients is None):
        #     logger.my_print('Fitting LR model directly using features: {}.'.format(features))
        # elif (submodels_features is None) and (lr_coefficients is not None):
        #     logger.my_print('Reusing coefficients of features: {} = {}.'.format(['intercept'] + features, lr_coefficients))
        # elif (submodels_features is not None) and (lr_coefficients is None):
        #     logger.my_print('Refitting submodels using features: {}.'.format(submodels_features))
        #     logger.my_print('Constructing final model from submodels using features: {}.'.format(features))
        # elif (submodels_features is not None) and (lr_coefficients is not None):
        # else:
        #     raise ValueError('Only one or both of submodels_features = {} (for fitting) and lr_coefficients = {} '
        #                      '(for reusing coefficients) should be None. If both are None, then we will directly fit '
        #                      'the logistic regression model on features = {}.'
        #                      .format(submodels_features, lr_coefficients, features))

        # Initialize variables
        model = LogisticRegression(random_state=seed, C=1e10)
        test_patient_ids, test_y_pred, test_y = [None] * 3

        # Only consider patients with non-missing value for `endpoint`, i.e., df.loc[:, endpoint] in `valid_endpoint_values`
        #df = df[df.loc[:, endpoint].isin(valid_endpoint_values)]

        # Update endpoint
        #df.loc[:, endpoint] = preprocess_label(df.loc[:, endpoint])

        # Load train_val_test_patient_ids.json for training-internal_validation-testing split
        patient_ids_dict_file = open(patient_ids_json)
        patient_ids_dict = json.load(patient_ids_dict_file)

        # Construct training-internal_validation-testing set
        df_train = df[df.loc[:, patient_id_col].isin(patient_ids_dict['train'])]
        df_train_masked = df_train[df_train.loc[:, endpoint].isin(valid_endpoint_values)].copy()
        df_val = df[df.loc[:, patient_id_col].isin(patient_ids_dict['val'])]
        if config.use_test_set:
            df_test = df[df.loc[:, patient_id_col].isin(patient_ids_dict['test'])]
        else:
            df_test = pd.DataFrame()

        # Construct endpoints
        train_y_masked = df_train_masked.loc[:, endpoint]
        n_train = len(train_y_masked)
        
        n_train_0 = sum(train_y_masked == 0)
        n_train_1 = sum(train_y_masked == 1)
        assert n_train == n_train_0 + n_train_1

        # only refit the models if no lr_coefficients are given
        if submodels_features is not None and lr_coefficients is None:
            # Initialize variables
            nr_submodels = len(submodels_features)
            #logger.my_print('Number of submodels: {}.'.format(nr_submodels))

            # Initialize list of submodel coefficients
            coeffs_list = []
            for i, features_i in enumerate(submodels_features):
                # Training set submodel i
                train_X_masked = df_train_masked.loc[:, features_i]

                # Fit submodel i
                model = model.fit(train_X_masked, train_y_masked)

                # Create dict of coefficients of submodel i
                keys = ['intercept'] + features_i
                values = np.append(model.intercept_, model.coef_)
                coeffs_i_dict = {k: v for (k, v) in zip(keys, values)}
                coeffs_list.append(coeffs_i_dict)

            #logger.my_print('coeffs_list: {}'.format(coeffs_list))

            # Construct coefficients of final model
            lr_coefficients = []
            for f in ['intercept'] + features:
                coeff_i = 0
                # For-loop over coefficients of submodels
                for d_i in coeffs_list:
                    if f in d_i.keys():
                        coeff_i += d_i[f]

                lr_coefficients.append(coeff_i / nr_submodels)

        #logger.my_print('lr_coefficients: {}.'.format(lr_coefficients))

        # Training set final model
        train_patient_ids = df_train_masked.loc[:, patient_id_col].tolist()
        train_X_masked = df_train_masked.loc[:, features]

        # Fit on training set or use pretrained coefficients
        model = model.fit(train_X_masked, train_y_masked)

        if lr_coefficients is not None:
            model.intercept_ = np.array([lr_coefficients[0]])
            model.coef_ = np.array([lr_coefficients[1:]])

        train_patient_ids = df_train.loc[:, patient_id_col].tolist()
        train_X = df_train.loc[:, features]
        train_y = df_train.loc[:, endpoint]

        train_y_pred = model.predict_proba(train_X)[:,1]
        train_y_pred = torch.tensor(train_y_pred, device=config.device)
        train_y = torch.tensor(train_y.tolist(), dtype=torch.int, device=config.device)

        # Internal validation set
        val_patient_ids = df_val.loc[:, patient_id_col].tolist()
        val_X = df_val.loc[:, features]
        val_y = df_val.loc[:, endpoint]
        # n_val = len(val_y)
        # n_val_0 = sum(val_y == 0)
        # n_val_1 = sum(val_y == 1)
        # n_val_2 = sum(val_y == -1)
        # assert n_val == n_val_0 + n_val_1

        
        
        val_y_pred = model.predict_proba(val_X)[:,1]
        val_y_pred = torch.tensor(val_y_pred, device=config.device)
        
        val_y = torch.tensor(val_y.tolist(), dtype=torch.int, device=config.device)

        # Print label distribution
        # logger.my_print('Training size (label=0): {}/{} ({}).'.format(n_train_0, n_train,
        #                                                               round(n_train_0 / n_train, nr_of_decimals)))
        # logger.my_print('Training size (label=1): {}/{} ({}).'.format(n_train_1, n_train,
        #                                                               round(n_train_1 / n_train, nr_of_decimals)))
        # logger.my_print('Internal validation size (label=0): {}/{} ({}).'.format(n_val_0, n_val,
        #                                                                          round(n_val_0 / n_val,
        #                                                                                nr_of_decimals)))
        # logger.my_print('Internal validation size (label=1): {}/{} ({}).'.format(n_val_1, n_val,
        #                                                                          round(n_val_1 / n_val,
        #                                                                                nr_of_decimals)))

        # Test set
        if len(df_test) > 0:
            test_patient_ids = df_test.loc[:, patient_id_col].tolist()
            test_X = df_test.loc[:, features]
            test_y = df_test.loc[:, endpoint]
            # n_test = len(test_y)
            # n_test_0 = sum(test_y == 0)
            # n_test_1 = sum(test_y == 1)
            # assert n_test == n_test_0 + n_test_1

            #test_y_pred = model.predict_proba(test_X)[:,1]
            test_y_pred = model.predict_proba(test_X)[:,1]
            test_y_pred = torch.tensor(test_y_pred, device=config.device)
            test_y = torch.tensor(test_y.tolist(), dtype=torch.int, device=config.device)
            # logger.my_print('Test size (label=0): {}/{} ({}).'.format(n_test_0, n_test,
            #                                                           round(n_test_0 / n_test, nr_of_decimals)))
            # logger.my_print('Test size (label=1): {}/{} ({}).'.format(n_test_1, n_test,
            #                                                           round(n_test_1 / n_test, nr_of_decimals)))

        train_y_pred_dict[endpoint] = train_y_pred
        train_y_dict[endpoint] = train_y
        val_y_pred_dict[endpoint] = val_y_pred
        val_y_dict[endpoint] = val_y
        test_y_pred_dict[endpoint] = test_y_pred
        test_y_dict[endpoint] = test_y
        learned_lr_coefficients[endpoint] = lr_coefficients

    return (train_patient_ids, train_y_pred_dict, train_y_dict,
            val_patient_ids, val_y_pred_dict, val_y_dict,
            test_patient_ids, test_y_pred_dict, test_y_dict,
            learned_lr_coefficients)













def run_logistic_regression_models(config, patient_id_col, logger):
    """
    Run logistic regression model on training (training + internal_validation) and test data.

    Note: if submodels_features and lr_coefficients are both None, then the logistic regression model will be
        directly fitted to features.

    Assumption: there are no overlapping of patient_ids between train, internal_validation and test set
        (see load_data.py).

    Args:
        config (Config): configuration object
        patient_id_col (str): name of the patient_id column
        submodels_features_list (list): list of lists of features for submodels
        features_list (list): list of features
        lr_coefficients_list (list): list of coefficients for submodels
        logger (Logger): logger object

    Returns:

    """   
    # submodels_features_list = config.submodels_features_list
    # features_lr_list = config.features_lr_list
    # lr_coefficients_list = config.lr_coefficients_list

    features_csv = os.path.join(config.data_dir, config.filename_stratified_sampling_test_csv)
    endpoint_list = config.endpoint_list
    patient_id_length = config.patient_id_length

    valid_endpoint_values = config.valid_endpoint_values
    seed = config.seed
    nr_of_decimals = config.nr_of_decimals

    patient_ids_json = os.path.join(config.exp_dir, config.filename_train_val_test_patient_ids_json)


    # Initialize variables
    test_y_pred_dict, test_y_dict = dict(), dict()

    # Load features + labels
    df = pd.read_csv(features_csv, sep=';', decimal='.')
    df[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df[patient_id_col]]

    learned_lr_coefficients = {}


    #df.loc[:, patient_id_col] = df.loc[:, patient_id_col].apply(lambda x: '%0.{}d'.format(patient_id_length) % int(x))

    for i, endpoint in enumerate(endpoint_list):
        logger.my_print('Fitting LR models for: {}.'.format(endpoint))
        
        endpoint_tox = endpoint.split("_")[0] # get only the name of the toxicty, not the endpoint time (e.g. only "Dysphagia", no "M06")
        # use that key to access the features lists. They are the same for all 'late' (post-treatment) CITOR models
        submodels_features = config.submodels_features_list_dict[endpoint_tox]
        features = config.features_lr_list_dict[endpoint_tox]
        lr_coefficients = config.lr_coefficients_list_dict[endpoint_tox]

        # Initialize variables
        model = LogisticRegression(random_state=seed, C=1e10)
        test_patient_ids, test_y_pred, test_y = [None] * 3

        # Only consider patients with non-missing value for `endpoint`, i.e., df.loc[:, endpoint] in `valid_endpoint_values`
        #df = df[df.loc[:, endpoint].isin(valid_endpoint_values)]

        # Update endpoint
        #df.loc[:, endpoint] = preprocess_label(df.loc[:, endpoint])

        # Load train_val_test_patient_ids.json for training-internal_validation-testing split
        patient_ids_dict_file = open(patient_ids_json)
        patient_ids_dict = json.load(patient_ids_dict_file)

        # Construct training-internal_validation-testing set
        df_train = df[df.loc[:, patient_id_col].isin(patient_ids_dict['train'])]
        df_train_masked = df_train[df_train.loc[:, endpoint].isin(valid_endpoint_values)].copy()
        df_val = df[df.loc[:, patient_id_col].isin(patient_ids_dict['val'])]
        if config.use_test_set:
            df_test = df[df.loc[:, patient_id_col].isin(patient_ids_dict['test'])]
        else:
            df_test = pd.DataFrame()

        # Construct endpoints
        train_y_masked = df_train_masked.loc[:, endpoint]
        n_train = len(train_y_masked)
        
        n_train_0 = sum(train_y_masked == 0)
        n_train_1 = sum(train_y_masked == 1)
        assert n_train == n_train_0 + n_train_1

        # Training set final model
        train_patient_ids = df_train_masked.loc[:, patient_id_col].tolist()
        train_X_masked = df_train_masked.loc[:, features]

        # Fit on training set or use pretrained coefficients
        model = model.fit(train_X_masked, train_y_masked)

        if lr_coefficients is not None:
            model.intercept_ = np.array([lr_coefficients[0]])
            model.coef_ = np.array([lr_coefficients[1:]])


        # Print label distribution
        # logger.my_print('Training size (label=0): {}/{} ({}).'.format(n_train_0, n_train,
        #                                                               round(n_train_0 / n_train, nr_of_decimals)))
        # logger.my_print('Training size (label=1): {}/{} ({}).'.format(n_train_1, n_train,
        #                                                               round(n_train_1 / n_train, nr_of_decimals)))
        # logger.my_print('Internal validation size (label=0): {}/{} ({}).'.format(n_val_0, n_val,
        #                                                                          round(n_val_0 / n_val,
        #                                                                                nr_of_decimals)))
        # logger.my_print('Internal validation size (label=1): {}/{} ({}).'.format(n_val_1, n_val,
        #                                                                          round(n_val_1 / n_val,
        #                                                                                nr_of_decimals)))

        # Test set
        test_patient_ids = df_test.loc[:, patient_id_col].tolist()
        test_X = df_test.loc[:, features]
        test_y = df_test.loc[:, endpoint]
        # n_test = len(test_y)
        # n_test_0 = sum(test_y == 0)
        # n_test_1 = sum(test_y == 1)
        # assert n_test == n_test_0 + n_test_1

        #test_y_pred = model.predict_proba(test_X)[:,1]
        test_y_pred = model.predict_proba(test_X)[:,1]
        test_y_pred = torch.tensor(test_y_pred, device=config.device)
        test_y = torch.tensor(test_y.tolist(), dtype=torch.int, device=config.device)
        # logger.my_print('Test size (label=0): {}/{} ({}).'.format(n_test_0, n_test,
        #                                                           round(n_test_0 / n_test, nr_of_decimals)))
        # logger.my_print('Test size (label=1): {}/{} ({}).'.format(n_test_1, n_test,
        #                                                           round(n_test_1 / n_test, nr_of_decimals)))

        test_y_pred_dict[endpoint] = test_y_pred
        test_y_dict[endpoint] = test_y
        learned_lr_coefficients[endpoint] = lr_coefficients

    return (test_patient_ids, test_y_pred_dict, test_y_dict,
            learned_lr_coefficients)
