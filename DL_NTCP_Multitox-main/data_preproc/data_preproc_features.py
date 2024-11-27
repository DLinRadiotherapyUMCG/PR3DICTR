import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_preproc_functions import Logger, create_folder_if_not_exists
import matplotlib.pyplot as plt


#import misc
import data_preproc_config as data_prc_cfg


def load_included_and_included_patients(spreadheet_path):
    """
    Loads the first two sheets in the CITOR_REDCAP file (i.e. the data, and the second tab of excluded patients)
    Joins them and returns one dataframe
    
    Args:
        spreadheet_path: the path to the CITOR_REDCAP excel spreadsheet
    Returns:
        df_dataset: a dataframe of the spreadsheet
    """
    df = []
    patient_IDs = []
    for sheet in ['CITOR_REDCAP_clinical_data_impo', 'Nog niet patiënten geexcludeerd']:
        data = pd.read_excel(spreadheet_path, sheet, index_col=None)
        data.index = [os.path.basename(sheet)] * len(data)

        data = data[~data['UMCG'].isin(patient_IDs)]
        patient_IDs += list(data['UMCG'])
        df.append(data)
        

    df_dataset = pd.concat(df)
    #df_dataset.drop_duplicates(subset=['UMCG'], inplace=True, keep='first')

    try:  df_dataset = df_dataset.drop(['Xerostomia_BSL'], axis=1)  # removes a column from the spreadsheet that breaks everything
    except: pass

    return df_dataset


def trim_dataset_column_names(df):
    """
    Adjusts the column names in the spreadsheet, generally makes them shorter (and nicer)
    Also translates some of them into English.
    
    Args:
        df: a dataframe (the REDCAP data)
    Returns:
        df: the same dataframe, but the column titles are processed and shortened
    """
    df.columns = df.columns.str.replace('HN35_', '')
    df.columns = df.columns.str.replace('DYSFAGIE_UMCGshort', 'Dysphagia')

    df.columns = df.columns.str.replace('UMCG', data_prc_cfg.patient_id_col)
    df.columns = df.columns.str.replace('GESLACHT', 'Sex')
    df.columns = df.columns.str.replace('LEEFTIJD', 'Age')
    df.columns = df.columns.str.replace('ROKEN', 'Smoking')
    df.columns = df.columns.str.replace('TSTAD_DEF', 'T_stage')

    return df


def load_dose_info(dose_info_file_path):
    """
    Loads a csv that contains all of the mean doses. 
    Returns a dataframe with only the patientID and (desired) mean dose values for each OAR
    
    Args:
        dose_info_file_path: a path to the csv file that contains all of the (mean) dose info for all of the OARs
    Returns:
        df_dose_means: a dataframe containing only the patientIDs and the desired dose feature columns
        dose_columns: the names of the dose columns kept, as a list of strings
    """
    if dose_info_file_path.endswith('.csv'):
        df_dose = pd.read_csv(dose_info_file_path, delimiter=';', index_col=None, decimal=",")
    elif dose_info_file_path.endswith('.xlsx'):
        df_dose = pd.read_excel(dose_info_file_path, index_col=None)
    else: 
        print("CANNOT OPEN DOSE FILE!")
        return None
    
    try: # patient ID column might need renaming
        df_dose = df_dose.rename(columns={"PatID": data_prc_cfg.patient_id_col, "PatientID": data_prc_cfg.patient_id_col})
    except:
        pass

    df_dose[data_prc_cfg.patient_id_col] = [str(item).zfill(data_prc_cfg.patient_id_length) for item in df_dose[data_prc_cfg.patient_id_col]]  # make sure that the patient IDs are strings of 7 digits (i.e. pad the small numbers with 0's)

    dose_columns = get_dose_column_names(df_dose)  # retrieve the names of the dose info we want to keep
    df_dose_means = df_dose[[data_prc_cfg.patient_id_col] + dose_columns]  # remove all excess columns

    return df_dose_means, dose_columns


def get_dose_column_names(df_dose):
    """
    Retrieves the names of the dose columns we want to keep
    Does so by seeing in the config file if we've named them explicitly (data_prc_cfg.dose_features), or not, and we want to
    scrape the columns of the dose csv file by a certain keyword (data_prc_cfg.meandose_target_word), it keeps all of the columns that contain that keyword
    If both are set to None in the config file, then return an empty list. 
    Priority is given to the explict list of names.
    
    Args:
        df_dose: the full dataframe of patient doses (for all OARs)
    Returns:
        dose_column_names: a list of strings, containing the column names we want to keep
    """
    if data_prc_cfg.dose_features is not None:            # explicit list of names
        dose_column_names = data_prc_cfg.dose_features
    elif data_prc_cfg.meandose_target_word is not None:   # scan and keep all that contain the target word (e.g. everything with 'meandose' in it)
        target_word = data_prc_cfg.meandose_target_word
        dose_column_names = [x for x in df_dose.columns if target_word in x]
    else:
        dose_column_names = []   # we don't want/need any dose info

    return dose_column_names
   
    


def load_data_spreadsheet(spreadheet_path, logger, use_excluded_patients_sheet=False):
    """
    Loads the CITOR_RECAP spreadsheet, and does some initial preprocessing:
    - opens the file
    - trims down the column names and anglicizes them
    - drops any duplicate patients, converts the PatientID column into strings (and not integers)
    - removes patients that are in the 'exclude_list'
    - removes columns that are not needed (that aren't clinical variables, patient IDs, or (desired) toxicity columns)
    
    Args:
        spreadheet_path: the path to the REDCAP data spreadsheet
        use_excluded_patients_sheet: (optional) whether to also load the excluded patients
    Returns:
        df_dataset: a dataframe with all patients, with all features (clinical, baseline, dose, endpoint), regardless of whether they're missing or not
        dose_columns: names of the dose columns
    """
    logger.my_print('Loading toxicity spreadsheet')
    if use_excluded_patients_sheet:
        logger.my_print('   also loading excluded patient sheet')
        df_dataset = load_included_and_included_patients(spreadheet_path)
    else:
        
        df_dataset = pd.read_excel(spreadheet_path, index_col=None)

    # adjust the column names
    df_dataset = trim_dataset_column_names(df_dataset)

    # convert PatientID column to a string dtype (also pad the short IDs with zeroes)
    df_dataset = df_dataset.sort_values(by=[data_prc_cfg.patient_id_col])
    df_dataset[data_prc_cfg.patient_id_col] = [str(item).zfill(data_prc_cfg.patient_id_length) for item in df_dataset[data_prc_cfg.patient_id_col]]
    # drop patients that should be excluded (according to the list in the config file)
    #df_dataset = df_dataset[~df_dataset['PatientID'].isin(data_prc_cfg.exclude_list)]

    logger.my_print(f'dropping patients that do not have images in {data_prc_cfg.save_dir_dataset_full}')
    # drop patients that we don't have images for
    patientIDs_with_images = next(os.walk(data_prc_cfg.save_dir_dataset_full))[1]
    df_dataset = df_dataset[df_dataset[data_prc_cfg.patient_id_col].isin(patientIDs_with_images)]

    # get all of the toxity baseline and endpoint columns to keep
    columns_to_keep = data_prc_cfg.clinical_features + data_prc_cfg.toxicity_baseline_columns + data_prc_cfg.toxicity_endpoint_columns   # also the clinical columns
    df_dataset = df_dataset[columns_to_keep]

    
    if (data_prc_cfg.dose_features is not None) or (data_prc_cfg.meandose_target_word is not None):
        logger.my_print('Loading dose info spreadsheet')
        # load the file containing all of the RT dose (e.g. mean dose per OAR):
        df_dose_means, dose_columns = load_dose_info(data_prc_cfg.dose_spreadsheet_file_path)
    
        # merge the doses onto df_dataset, by patientID (do not add new patients)
        df_dataset = df_dataset.merge(df_dose_means, how='left', on=data_prc_cfg.patient_id_col)
    else: 
        logger.my_print('Skipped dose info')

    logger.my_print(f'Adding CT+C column, based on the folder types in the dataset (from {data_prc_cfg.filename_patient_folder_types_csv})')
    df_dataset = add_CT_C_column(df_dataset, logger)

    return df_dataset





def add_CT_C_column(df1, logger):
    """
    
    """
    folder_types_csv_dir = os.path.join(data_prc_cfg.save_root_dir, data_prc_cfg.filename_patient_folder_types_csv)
    #print(folder_types_csv_dir)
    df2 = pd.read_csv(folder_types_csv_dir, delimiter=';', index_col=None)
    df2.rename(columns={"Patient_id": data_prc_cfg.patient_id_col}, inplace=True)
    #df2[data_prc_cfg.patient_id_col] = [str(item).zfill(data_prc_cfg.patient_id_length) for item in df2[data_prc_cfg.patient_id_col]]

    df2[data_prc_cfg.patient_id_col] = df2[data_prc_cfg.patient_id_col].astype(str).str.zfill(7)
    df2["CT+C"] = df2["Folder"].apply(lambda x: 1 if x == "with_contrast" else 0)
    df2 = df2[[data_prc_cfg.patient_id_col, 'CT+C']]
    
    df = pd.merge(df1, df2, on=data_prc_cfg.patient_id_col, how='inner')

    return df




def main():
    
    logger = Logger(output_filename=None)

    # load the REDCAP spreadsheet (i.e. load the tox baselines and endpoints)
    # use_excluded_patients_sheet=True because one patient that has images, is in the excluded list
    df_toxicity_dataset = load_data_spreadsheet(data_prc_cfg.toxicity_endpoints_spreadsheet_file_path, logger, use_excluded_patients_sheet=True)
    logger.my_print('Saving patients_all_features spreadsheets to final dataset directory')
    dir_patient_features_csv = os.path.join(data_prc_cfg.save_dir_final_dataset, data_prc_cfg.filname_features_spreadsheet_csv)
    dir_patient_features_excel = os.path.join(data_prc_cfg.save_dir_final_dataset, data_prc_cfg.filname_features_spreadsheet_excel)
    df_toxicity_dataset.to_csv(dir_patient_features_csv, sep=';', index=None)
    df_toxicity_dataset.to_excel(dir_patient_features_excel, index=None)

    logger.my_print('DONE data_preproc_features.py')
    logger.close()
    del logger


if __name__ == '__main__':
    main()
    

