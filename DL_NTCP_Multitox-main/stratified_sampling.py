import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_preproc'))  # prevents an error when importing 'data_preproc.data_preproc_config'

import numpy as np
import pandas as pd
import data_preproc.data_preproc_config as preprc_cfg
import config as cfg
from data_preproc.data_preproc_functions import Logger
from data_preproc.data_preproc_functions import get_all_combinations

patient_id_col = preprc_cfg.patient_id_col    # column name of patientIDs


def load_all_features_csv():
    """
    Loads the csv file that contains all recorded features, for all patients.
    Args:
        
    Returns: 
        df_features: a pd dataframe with all of the features we have for all of the patients.
    """

    # load the csv
    features_csv_path = os.path.join(cfg.data_dir, cfg.filename_features_csv)
    print(features_csv_path)
    df_features = pd.read_csv(features_csv_path, delimiter=';', index_col=None)

    print(df_features.columns.tolist())
    # make sure the patient IDs are the correct length
    df_features[patient_id_col] = [str(item).zfill(cfg.patient_id_length) for item in df_features[patient_id_col]]

    return df_features


def remove_unnecessary_columns(df):
    """
    Removes all feature columns that are not used in the main config file.
    Args:
        df: dataframe with all feature columns in it.
    Returns: 
        df: dataframe with only the columns specified by the config file.
    """
    column_names = cfg.main_columns + cfg.toxicity_baseline_features + cfg.toxicity_endpoint_features
    df = df[column_names]

    return df



def remove_patients(df, patients_to_remove):
    """
    Removes the patients that should be excluded. The excluded patients are read from the excluded_patients_umcg.csv, as well as a list in the main config file.
    Args:
        df: a dataframe.
    Returns: 
        df: the same dataframe, but without the excluded patients.
    """
    df = df[~df[patient_id_col].isin(patients_to_remove)].copy()

    return df

def remove_patients_missing_endpoints(df):
    """
    Removes the patients that do not have at least 1 endpoint. 
    Args:
        df: a dataframe.
    Returns: 
        df: the same dataframe, but only with patients that have at least 1 endpoint
    """
    for col in cfg.toxicity_endpoint_features:
        num_dropped = len(df) - len(df.dropna(subset=[col]))
        if num_dropped > 0:
            print(f"    {num_dropped} patients because of missing values in column {col}")
            #cols_with_missing_values.append(cols_with_missing_values)
    df = df.dropna(subset=cfg.toxicity_endpoint_features, how='all')

    return df


def keep_patients_with_all_data_present(df, specific_cols = None):
    """
    Makes a df that consists only of patients that have all input and output features present (i.e. that are not missing any data in the features specified in the main config file).
    Args:
        df: a dataframe.
    Returns: 
        df_filtered: the same dataframe, but without any missing values.
    """
    cols_with_missing_values = []
    for col in specific_cols:
        num_dropped = len(df) - len(df.dropna(subset=[col]))
        if num_dropped > 0:
            print(f"Dropping {num_dropped} patients because of missing values in column {col}")
            cols_with_missing_values.append(col)
    #print(df.columns.tolist())
    print(specific_cols)
    if specific_cols == None:
        df_filtered = df.dropna()
    else:
        df_filtered = df.dropna(subset=specific_cols)  # drop all patients with missing values
    num_dropped = len(df) - len(df_filtered)
    print(f'Dropped {num_dropped} patients because of missing values in the following columns: {cols_with_missing_values}')

    return df_filtered




def stratified_sampling_split(df, frac, logger, name):
    """
    Creates a train_test set of the given dataframe. The size of the test set is frac*total_number_of_patients_in_dataset, so that the test set's size is proportional to the entire useable dataset.
    Does stratified sampling based on the columns defined in the config file. 
    Args:
        df: a dataframe
        frac: % size of test set, relative to the total size of the dataset
        total_number_of_patients_in_dataset: total size of the dataset = number of patients that are useable for the DL model (i.e. that have all input features, but might miss some endpoints).
    Returns: 

    """    
    n_patients = len(df)
    n_samples_testing_set = int(frac * n_patients)

    logger.my_print(f'Making {name} set. Size = {frac} ({n_samples_testing_set} patients)')
    logger.my_print(f'Stratifed sampling based on columns: {cfg.strata_groups}')

    df_test = df.groupby(cfg.strata_groups, group_keys=False, dropna=False).apply(lambda x: x.sample(frac=frac, replace=False,
                                                                                random_state=cfg.seed))
    df_train = df.loc[~df.index.isin(df_test.index)]

    test_set_length_difference = n_samples_testing_set - len(df_test)
    
    # if the test set is not the exact size it should be, address this by resampling from either the train or test sets (whichever is too large)
    while test_set_length_difference != 0:  
        if test_set_length_difference > 0:   # test set is too small, pull some random datapoints from the training set
            logger.my_print(f"Stratified sampling returned too small of a test set ({len(df_test)}). Randomly sampling {test_set_length_difference} samples from the training set.")
            try:
                rows = df_train.groupby(cfg.strata_groups, group_keys=False, dropna=False).apply(lambda x: x.sample(n=test_set_length_difference, replace=False, random_state=cfg.seed))
            except:
                rows = df_train.apply(lambda x: x.sample(n=test_set_length_difference, replace=False, random_state=cfg.seed))
            
            rows_index = rows.index
            df_test = pd.concat([df_test, rows])
            df_train = df_train.drop(rows_index)
                             
        else:     # test set is too large, move some random datapoints to the training set
            test_set_length_difference *= -1
            logger.my_print(f"Stratified sampling returned too large of a test set ({len(df_test)}). Moving {test_set_length_difference} samples back to the training set.")
            try:
                rows = df_test.groupby(cfg.strata_groups, group_keys=False, dropna=False).apply(lambda x: x.sample(n=test_set_length_difference, replace=False, random_state=cfg.seed))
            except:
                rows = df_test.apply(lambda x: x.sample(n=test_set_length_difference, replace=False, random_state=cfg.seed))

            rows_index = rows.index
            df_train = pd.concat([df_train, rows])
            df_test = df_test.drop(rows_index)

        test_set_length_difference = n_samples_testing_set - len(df_test)

    test_patient_IDs = list(df_test[preprc_cfg.patient_id_col])
    df_train_val = remove_patients(df, patients_to_remove=test_patient_IDs)
    
    return df_train_val, df_test


def preprocess_clinical_variables(df):
    """
    Normalises age, and encodes the 'Sex' column into integer values
    Args:
        df: the features dataframe
    Returns:
        df: the same dataframe, but with the Age and Sex columns preprocessed
    """
    # normalise age by dividing by 100
    df['Age'] = df['Age'] / 100
    
    # binary encode Sex
    df['Sex'] = df['Sex'].replace({"Vrouw":int(0), "Man":int(1)})
    df.head(-10)

    return df





def binarise_basline_toxicity_columns(df, tox_cols_to_binarise):
    """
    binarises the toxicity columns of a dataframe, and makes new binary-encoded columns for each toxicity level (e.g. "Helemaal niet", "Een beetje", etc), that can be used for the DL model baseline values input
    
    Args:
        df: the features dataframe
        toxicity_columns: the column names for all of the toxicity baselines and endpoints
    Returns:
        df: the same as the input df, but the toxicity columns are now binary, and new binary columns have been added in the format of the baseline input to the DL model
    """
    # add underscores to the toxicity values
    value_mapping = {"Nogal":"Nogal", "Heel erg":"Heel_erg",  
                     "Een beetje":"Een_beetje", "Helemaal niet":"Helemaal_niet" } 
    df = df.replace(value_mapping)

    for tox_col_name in tox_cols_to_binarise:
        if "Dysphagia" in tox_col_name:
            dysphagia_mapping = {
                "geen slikklachten/normaal eetpatroon / wel slikklachten, maar kan wel alles eten": "Grade0_1",
                "geen slikklachten/normaal eetpatroon": "Grade0",
                "alleen gepureerd/zacht voedsel": "Grade2",
                "wel slikklachten, maar kan wel alles eten": "Grade1",
                "neussonde of PEG, maar nog wel orale intake mogelijk / neussonde of PEG, geen enkele orale intake meer mogelijk": "Grade3_4",
                "alleen vloeibaar voedsel, geen sondevoeding": "Grade3",
                "neussonde of PEG, maar nog wel orale intake mogelijk": "Grade4",
                "neussonde of PEG, geen enkele orale intake meer mogelijk": "Grade5",
            }
            df = df.replace(dysphagia_mapping)

            dysphagia_mapping_step2 = {
                "Grade0"   : "Grade0_1",
                "Grade0_1" : "Grade0_1",
                "Grade1"   : "Grade0_1",
                "Grade2"   : "Grade2",
                "Grade3"   : "Grade3_4",
                "Grade4"   : "Grade3_4",
                "Grade4_5" : "Grade3_4",
                "Grade5"   : "Grade3_4",
            }
            df = df.replace(dysphagia_mapping_step2)
        

        # Binarize the values in 'tox_col_name'
        df_binary_columns = pd.get_dummies(df[tox_col_name], prefix=tox_col_name)
        df_binary_columns = df_binary_columns.astype('Int64')
        df_binary_columns = add_missing_tox_baseline_binary_columns(df_binary_columns, tox_col_name)
        if "Dysphagia" not in tox_col_name:
            # make a new column that joins the two highest levels of toxicity ("Nogal" and "Heel erg") together
            df_binary_columns[f"{tox_col_name}_Nogal_Heel_erg"] = df_binary_columns[f"{tox_col_name}_Nogal"] + df_binary_columns[f"{tox_col_name}_Heel_erg"]
            
        # Concatenate the binarized columns with the original DataFrame
        df = pd.concat([df, df_binary_columns], axis=1)

    # binarise the original columns
    value_mapping = {"Grade0_1":int(0), "Grade2":int(1), "Grade3_4":int(1)}  ## for dysphagia
    df = df.replace(value_mapping)
    #value_mapping = {"not_at_all":0, "little":0, "moderate":1, "severe":1}
    value_mapping = {"Helemaal_niet":int(0), "Een_beetje":int(0), "Nogal":int(1), "Heel_erg":int(1)}
    df = df.replace(value_mapping)
    
    return df



def add_missing_tox_baseline_binary_columns(df, tox_col_name):
    """
    In case no patients have one of the 4 endpoints (e.g. if using only a handful of patients, this is possible),
    then make a new column for that endpoint, and set all values to 0.
    """
    if f"{tox_col_name}_Nogal" not in df.columns:
        df[f"{tox_col_name}_Nogal"] = 0
    if f"{tox_col_name}_Heel_erg" not in df.columns:
        df[f"{tox_col_name}_Heel_erg"] = 0
    if f"{tox_col_name}_Een_beetje" not in df.columns:
        df[f"{tox_col_name}_Een_beetje"] = 0
    if f"{tox_col_name}_Helemaal_niet" not in df.columns:
        df[f"{tox_col_name}_Helemaal_niet"] = 0

    return df


def add_missing_binary_feature_columns(df, cols_to_binarise):
    for col_name in cols_to_binarise:
        if col_name not in df.columns:
            df[col_name] = 0
    return df


def binarise_feature_columns(df, cols_to_binarise):
    """
    binarises the toxicity columns of a dataframe, and makes new binary-encoded columns for each toxicity level (e.g. "Helemaal niet", "Een beetje", etc), that can be used for the DL model baseline values input
    
    Args:
        df: the features dataframe
        toxicity_columns: the column names for all of the toxicity baselines and endpoints
    Returns:
        df: the same as the input df, but the toxicity columns are now binary, and new binary columns have been added in the format of the baseline input to the DL model
    """

    #value_mapping = {"Nogal":"Nogal_Heel_erg", "Heel erg":"Nogal_Heel_erg",  # join together the two highest levels of toxicity
    #                 "Een beetje":"Een_beetje", "Helemaal niet":"Helemaal_niet" }  # rename the other two as well (to get rid of the spaces in the column name)
    #df = df.replace(value_mapping)

    for col_name in cols_to_binarise:
        if "Loctum2" in col_name:
            loctum_mapping = {
                'Categorie: overig' : "Overig",
                'Hypofarynx' : "Pharynx",
                'Larynx' : "Larynx",
                'Nasopharynx' : "Pharynx",
                'Neus(bij)holte' : "Overig",
                'Oral Cavity' : "Oral_Cavity",
                'Oropharynx' : "Pharynx"}
            df[col_name] = df[col_name].replace(loctum_mapping)

        #levels=c("Categorie: overig", "Hypofarynx", "Larynx", "Nasopharynx",  "Neus(bij)holte", "Oral Cavity",  "Oropharynx" ), #<-- levels are the original names in the DATA
        #labels=c("Overig",            "Pharynx",    "Larynx", "Pharynx",       "Overig",        "Oral Cavity",   "Pharynx"))  
        elif col_name == "WHO":
            WHO_mapping = {
                "WHO 0"   : "0",
                "WHO 0-1" : "1",
                "WHO 1"   : "1",
                "WHO 1-2" : "2",
                "WHO 2"   : "2",
                "WHO 3"   : "3",
                "WHO 4"   : "4"
            }
            df = df.replace(WHO_mapping)

        elif col_name == "Smoking":
            smoking_mapping = {
                "Ja, in verleden" : int(0),   # ! Ja, inverleden becomes a 0 label !
                "Ja, nog steeds" : int(1),
                "Nee" : int(1)
            }
            df[col_name] = df[col_name].replace(smoking_mapping)
        elif col_name == "P16":
            P16_mapping = {
                "Niet bepaald" : "Niet_bepaald",
                "Positief" : "Positief",
                "Negatief" : "Negatief",
            }
            df[col_name] = df[col_name].replace(P16_mapping)
        elif col_name == "T_stage":
            T_stage_mapping = {
                "Tx" : "Tx",
                "Tis" : "Tis",
                "T0" : "T0",
                "T1" : "T1",
                "T1a" : "T1",
                "T1b" : "T1",
                "T2" : "T2",
                "T3" : "T3",
                "T4" : "T4",
                "T4a" : "T4",
                "T4b" : "T4"
            }
            df[col_name] = df[col_name].replace(T_stage_mapping)
            #df.rename(columns={"T_stage": ""}, inplace=True)
        elif col_name == "N_stage":
            N_stage_mapping = {
                "Nx" : "Nx",
                "N0" : "N0",
                "N1" : "N1",
                "N2" : "N2",
                "N2a" : "N2",
                "N2b" : "N2",
                "N2c" : "N2",
                "N3" : "N3",
                "N3a" : "N3",
                "N3b" : "N3",
                "N3c" : "N3"
            }
            df[col_name] = df[col_name].replace(N_stage_mapping)

        # Binarize the values in 'tox_col_name'
        df_binary_columns = pd.get_dummies(df[col_name], prefix=col_name)
        df_binary_columns = df_binary_columns.astype('Int64')

        # make the T-stage and N-stage column names shorter
        df_binary_columns.columns = df_binary_columns.columns.str.replace('T_stage_', '')
        df_binary_columns.columns = df_binary_columns.columns.str.replace('N_stage_', '')

        #df_binary_columns = add_missing_binary_feature_columns(df_binary_columns, [col_name])

        # Concatenate the binarized columns with the original DataFrame
        df = pd.concat([df, df_binary_columns], axis=1)
    
    return df


def replace_missing_values(df):
    """
    
    Args:
        
    Returns: 

    """
    #value_mapping = {None:int(-1)}  ## for dysphagia
    #df = df.replace(value_mapping)
    df = df.fillna(int(-1))
    return df


def replace_missing_W01_values(df):
    """
    
    Args:
        
    Returns: 

    """

    WO1_cols = get_all_combinations(cfg.target_toxicities, ["W01"], spacer='_')  # W01
    BSL_cols = get_all_combinations(cfg.target_toxicities, ["BSL"], spacer='_')  # BSL

    for tox_BSL_col, tox_W01_col in zip(BSL_cols, WO1_cols):
        df[tox_W01_col] = df[tox_W01_col].fillna(df[tox_BSL_col])
    
    return df







def load_and_prepare_total_dataset(logger, suppress_CLI_prints=False):
    # first, load the csv of all 1168 patients
    df_features = load_all_features_csv()
    if not suppress_CLI_prints:
        logger.my_print(f'Total number of patients: {len(df_features)}')
    # remove the excluded patients from that
    df_features = remove_patients(df_features, patients_to_remove=cfg.exclude_list)

    if not suppress_CLI_prints:
        logger.my_print(f'Number of patients after removing excluded patients: {len(df_features)}')
    
    # first fill missing W01 values if BSL values
    df_features = replace_missing_W01_values(df_features)
    
    # drop all of the feature columns that we don't need
    #df_features = remove_unnecessary_columns(df_features)
    if not suppress_CLI_prints:
        logger.my_print(f'features loaded: {df_features.columns.to_list()}')

    # drop patients that don't have at least 1 endpoint
    len_df_features = len(df_features)
    df_dataset = remove_patients_missing_endpoints(df_features)
    N_patients_with_endpoint = len(df_dataset)
     #print(total_number_of_patients_in_dataset)
    if not suppress_CLI_prints:
        logger.my_print(f"Number of patients dropped for having no endpoint: {len_df_features - N_patients_with_endpoint}")
        logger.my_print(f'Number of patients after removing those that do not have any endpoint: {N_patients_with_endpoint}')
    

    # drop patients that are missing any of the models' input
    required_features = cfg.all_input_features 
    if not suppress_CLI_prints:
        logger.my_print(f'Keeping only patients with all of the following features: {required_features}')
    df_dataset = keep_patients_with_all_data_present(df_dataset, specific_cols=required_features)
    N_patients_has_inputs_and_endpoint = len(df_dataset)

    if not suppress_CLI_prints:
        logger.my_print(f'Number of patients dropped for not having all inputs present: {N_patients_with_endpoint - N_patients_has_inputs_and_endpoint}')
        logger.my_print(f'Number of patients with all inputs present: {N_patients_has_inputs_and_endpoint}')


    return df_dataset











def make_stratified_test_set(df_dataset, suppress_CLI_prints=False):
    """
    
    Args:
        
    Returns: 

    """
    
    
    #new_test_frac = (cfg.test_frac * len(df_dataset)) / len(df_dataset)
    _, df_test = stratified_sampling_split(df=df_dataset, frac=cfg.test_frac, logger=logger, name='test')
    test_patient_IDs = list(df_test[preprc_cfg.patient_id_col])

    # drop the test patients from the train-validate set
    # we can use all patients for this, even if they're missing some endpoints
    df_train_val = remove_patients(df_dataset, patients_to_remove=test_patient_IDs)

    # make a validation set
    #df_val, val_patient_IDs = stratified_sampling_split(df=df_train_val, frac=cfg.val_frac, logger=logger, name='validation')
    #df_train = remove_patients(df_train_val, patients_to_remove=val_patient_IDs)

    # ensure that no data leakage is happening (i.e. no overlap of patients between these sets)
    patient_id_col = preprc_cfg.patient_id_col
    assert len(df_dataset) == len(df_train_val[patient_id_col].tolist() + df_test[patient_id_col].tolist())
    assert len(df_dataset) == len(set(df_train_val[patient_id_col].tolist() + df_test[patient_id_col].tolist()))

    # Create column indicating whether the patient_id belongs to 'train', 'val' or 'test'
    df_train_val[cfg.split_col] = 'train_val'
    df_test[cfg.split_col] = 'test'

    # rejoin these dataframes
    df_split = pd.concat([df_train_val, df_test]).sort_index()
    
    # binarise the features (for DL model)
    if not suppress_CLI_prints:
        logger.my_print('Binarising (non-dose) feature columns')
    df_split = preprocess_clinical_variables(df_split)  
    df_split = binarise_feature_columns(df_split, cols_to_binarise = ["Loctum2", "T_stage", "N_stage", "Smoking", "P16", "WHO"] )  
    df_split = binarise_basline_toxicity_columns(df_split, tox_cols_to_binarise = cfg.toxicity_baseline_features)

    # replace any None values with a -1
    if not suppress_CLI_prints:
        logger.my_print('Encode missing values as -1') 
    df_split = replace_missing_values(df_split)  

    
    df_split[patient_id_col] = df_split[patient_id_col].astype(str)
    dir_df_split = os.path.join(cfg.data_dir, cfg.filename_stratified_sampling_test_csv)
    logger.my_print(f"Saving stratified sampling csv to: '{dir_df_split}'")
    df_split.to_csv(dir_df_split, sep=';', index=False)

    return df_split


def save_strat_sampling_test_df(df, logger):
    df[patient_id_col] = df[patient_id_col].astype(str)
    dir_df_split = os.path.join(cfg.data_dir, cfg.filename_stratified_sampling_test_csv)
    logger.my_print(f"Saving stratified sampling csv to: '{dir_df_split}'")
    df.to_csv(dir_df_split, sep=';', index=False)


if __name__ == '__main__':
    suppress_CLI_prints = False
    logger = Logger(output_filename = 'stratified_sampling.log', suppress_CLI_prints=suppress_CLI_prints)

    df_dataset = load_and_prepare_total_dataset(logger, suppress_CLI_prints=suppress_CLI_prints)

    df = make_stratified_test_set(df_dataset, suppress_CLI_prints=False)
    save_strat_sampling_test_df(df, logger)


    per_tox_subfolder = os.path.join(cfg.data_dir, "per_toxicity_csv_")  + f"{cfg.seed}"
    os.makedirs(per_tox_subfolder, exist_ok=True)
    # Iterate over each column name in the list
    
    for column_name in cfg.toxicity_endpoint_features:
        # Drop rows with -1 in the current column
        df_copy = df[df[column_name] != -1].copy()

        # Save the resulting dataframe
        df_copy.to_csv(os.path.join(per_tox_subfolder, f'{column_name}_{cfg.seed}.csv'), index=False, sep=";")

    
    
    # for i in range(501,1001):
    #     cfg.seed = i
    
    #     df = make_stratified_test_set(df_dataset, suppress_CLI_prints=True)

    #     per_tox_subfolder = os.path.join(cfg.data_dir, "per_toxicity_csvs") 
    #     os.makedirs(per_tox_subfolder, exist_ok=True)
    #     # Iterate over each column name in the list
        
    #     for column_name in cfg.toxicity_endpoint_features:
    #         # Drop rows with -1 in the current column
    #         df_copy = df[df[column_name] != -1].copy()

    #         # Save the resulting dataframe
    #         df_copy.to_csv(os.path.join(per_tox_subfolder, f'{column_name}_{cfg.seed}.csv'), index=False, sep=";")
    
   