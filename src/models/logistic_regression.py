import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression

from src.constants import PATIENT_ID_COL_NAME, PATIENT_ID_LENGTHS_DICT, EARLY_NTCP_TIMEPOINTS, LATE_NTCP_TIMEPOINTS



def determine_CITOR_model_name(endpoint_name):
    """
    Helper function to determine the CITOR model name based on the endpoint name 
    (i.e. work out if the desired timepoint is an early or a late timepoint).
    Helps to figure out which model features to read from the config.
    """
    endpoint_name = endpoint_name.lower()

    toxicity_name = "_".join(endpoint_name.split("_")[:-1])
    if toxicity_name == 'sticky':
        toxicity_name = 'sticky_saliva'

    timepoint = endpoint_name.split("_")[-1]

    CITOR_name = toxicity_name
    if timepoint in EARLY_NTCP_TIMEPOINTS:
        CITOR_name += "_early"
    else:
        CITOR_name += "_late"

    return CITOR_name



def refit_CITOR_model(config, df_train, endpoint = "Xerostomia_M06", CITOR_model_name = 'xerostomia_late'):
    """
    A function that refits the CITOR model to a training set
    """
    
    if len(config['CITOR'][CITOR_model_name]['models'].keys()) > 1:
        feature_dict = {'model1':{}, 'model2':{}}
        submodel_names = ['model1', 'model2']

        #all_feature_columns = list(set()) 
    else:
        feature_dict = {'model1':{}}
        submodel_names = ['model1']   

    # refit (all of the sub-) models
    for submodel_names in submodel_names:
        # select the columns per submodel
        feature_column_names = np.array(config['CITOR'][CITOR_model_name]['models'][submodel_names]['features'])

        X_train = df_train[feature_column_names]
        y_train = df_train[endpoint]

        # make and fit the (sub)model
        sub_model = LogisticRegression(random_state=config['general']['seed'], C=1e10, max_iter = 1000)

        # Messy things to handle any missing data
        valid_indices = y_train != -1
        X_train = X_train[valid_indices]
        y_train = y_train[valid_indices]
        
        # fit the model
        sub_model.fit(X_train, y_train)

        # save the submodel in the feature_dict
        feature_dict[submodel_names]['features'] = feature_column_names
        feature_dict[submodel_names]['coef'] = sub_model.coef_[0]
        feature_dict[submodel_names]['intercept'] = sub_model.intercept_[0]

        # print(f"submodel {submodel_names} has been fitted")
        # sub_model_dict = {f: c for f, c in zip(feature_column_names, sub_model.coef_[0])}
        # print(sub_model_dict)
        # print(feature_dict[submodel_names]['intercept'])


    # combine the submodels
    if len(feature_dict.keys()) > 1:
        model1 = list(feature_dict.keys())[0]
        model2 = list(feature_dict.keys())[1]
        feature_names = []
        coef = []
        feature_names.append('intercept')
        coef.append(.5*(feature_dict[model1]['intercept'] + feature_dict[model2]['intercept']))
        
        features1 = feature_dict[model1]['features']
        features2 = feature_dict[model2]['features']
        
        overlap =  list(set(features1) & set(features2))
        remainder_m1 = list(set(features1) - set(features2))
        remainder_m2 = list(set(features2) - set(features1))

        
        for i in range(len(remainder_m1)):
            feature_names.append(remainder_m1[i])
            coef.append(.5*(feature_dict[model1]['coef'][np.where(features1 == remainder_m1[i])])[0])         
        for i in range(len(remainder_m2)):
            feature_names.append(remainder_m2[i])
            coef.append(.5*(feature_dict[model2]['coef'][np.where(features2 == remainder_m2[i])])[0])    
        for i in range(len(overlap)):
            feature_names.append(overlap[i])
            coef.append(.5*(feature_dict[model1]['coef'][np.where(features1 == overlap[i])]+feature_dict[model2]['coef'][np.where(features2 == overlap[i])])[0])     
   
    else:
        feature_names = ['intercept'] + list(feature_column_names)
        coef = [sub_model.intercept_[0]] + list(sub_model.coef_[0]) 


    # print(features)
    # features_coeff_dict = {k: v for k, v in zip(features, coef)}
    # print(features_coeff_dict)
    # print(coef[1:])
    
    model = LogisticRegression(random_state=config['general']['seed'], C=1e10, max_iter = 1000, multi_class='ovr')
    model.intercept_ = np.array([coef[0]])
    model.coef_ = np.array([coef[1:]])

    return model, feature_names
    












































from src.utils.saving.saving_predictions import concatenate_predictions, save_predictions


def determine_CITOR_model_name(endpoint_name):
    endpoint_name = endpoint_name.lower()

    toxicity_name = "_".join(endpoint_name.split("_")[:-1])
    if toxicity_name == 'sticky':
        toxicity_name = 'sticky_saliva'

    timepoint = endpoint_name.split("_")[-1]

    CITOR_name = toxicity_name
    if timepoint in EARLY_NTCP_TIMEPOINTS:
        CITOR_name += "_early"
    else:
        CITOR_name += "_late"

    return CITOR_name




from src.evaluation.total_evaluation import total_evaluation_current_fold
from src.evaluation.get_visualisations import get_visualizations


def run_logistic_regression(config, df_features, train_patient_IDs, val_patient_IDs, test_patient_IDs=[]):
    # get the train and val patients' data
    df_train = df_features[df_features[PATIENT_ID_COL_NAME].isin(train_patient_IDs)]
    df_val = df_features[df_features[PATIENT_ID_COL_NAME].isin(val_patient_IDs)]
    assert len(df_train) == len(train_patient_IDs)
    assert len(df_val) == len(val_patient_IDs)

    # get the test patients' data (if you want that)
    if len(test_patient_IDs) > 0:
        df_test = df_features[df_features[PATIENT_ID_COL_NAME].isin(test_patient_IDs)]
        assert len(df_test) == len(test_patient_IDs)
    else:
        df_test = None

    sets = ['train']
    if len(val_patient_IDs) > 0:
        sets.append('val')
    if len(test_patient_IDs) > 0:
        sets.append('test')
    
    endpoint_list = config['columns']['labels']

    all_preds_dict = dict()
    all_targets_dict = dict()

    for idx, endpoint in enumerate(endpoint_list):
        #print(f"fitting models for {endpoint}")

        # get the features and labels
        CITOR_model_name = determine_CITOR_model_name(endpoint)

        model, feature_column_names = refit_CITOR_model(config, df_train, endpoint = endpoint, CITOR_model_name = CITOR_model_name)

        #print("\n\n")
                         # endpoint, train_patient_IDs, val_patient_IDs, test_patient_IDs)

        # get the predictions
        df_train_X = df_train.loc[:, feature_column_names].values
        df_val_X = df_val.loc[:, feature_column_names].values

        train_patient_IDs = list(df_train[PATIENT_ID_COL_NAME].values)
        val_patient_IDs = list(df_val[PATIENT_ID_COL_NAME].values)

        train_preds = model.predict_proba(df_train_X)[:,1]
        val_preds = model.predict_proba(df_val_X)[:,1]

        if test_patient_IDs is not None:
            df_test_X = df_test.loc[:, feature_column_names].values
            test_patient_IDs = list(df_test[PATIENT_ID_COL_NAME].values)
            test_preds = model.predict_proba(df_test_X)[:,1]
            # store the predictions
            all_preds_dict[endpoint] = np.concatenate([train_preds, val_preds, test_preds])
            all_targets_dict[endpoint] = np.concatenate([df_train[endpoint].values, df_val[endpoint].values, df_test[endpoint].values])
        # no test set
        else:
            df_test_X = None
            test_patient_IDs = []
            test_preds = None
            # store the predictions
            all_preds_dict[endpoint] = np.concatenate([train_preds, val_preds])
            all_targets_dict[endpoint] = np.concatenate([df_train[endpoint].values, df_val[endpoint].values])
        

    all_patientIDs_list = train_patient_IDs + val_patient_IDs + test_patient_IDs
    mode_list = ["train"] * len(train_patient_IDs) + ["val"] * len(val_patient_IDs) + ["test"] * len(test_patient_IDs)

    # save all the predictions into one csv file
    save_predictions(config, all_patientIDs_list, all_preds_dict, all_targets_dict, mode_list, filename="all_lr_predictions.csv")
    

    pred_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], "all_lr_predictions.csv")
    total_evaluation_current_fold(config, sets = sets, pred_csv_dir = pred_csv_dir, lr=True)

    get_visualizations(config, sets = sets, pred_csv_dir = pred_csv_dir, lr=True)

    
    
    

