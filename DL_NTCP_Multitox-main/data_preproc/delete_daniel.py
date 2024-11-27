import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from monai.transforms import Compose, Resize

import data_preproc_config as data_prc_cfg
from data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from data_preproc_ct_segmentation_map import spacing_correction


"""
FROM DATA_PREPROC.PY
"""


def main_dataset():
    """
    Copy-paste .npy files to dataset folder.

    Returns:

    """
    # Initialize variables
    use_umcg = data_prc_cfg.use_umcg
    mode_list = data_prc_cfg.mode_list
    data_dir_mode_list = data_prc_cfg.save_dir_mode_list
    patient_id_length = data_prc_cfg.patient_id_length

    save_root_dir = data_prc_cfg.save_root_dir
    save_dir_dataset_full = data_prc_cfg.save_dir_dataset_full
    # TODO: temporary
    save_dir_dataset = data_prc_cfg.save_dir_dataset
    # Taste
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/MDACC/preprocessed/dataset_full_mdacc'
    # save_dir_dataset = 'D:/MyFirstData_MDACC/dataset_mdacc_taste'
    # data_prc_cfg.filename_endpoints_csv = 'taste_features_MDACC.csv'
    # data_prc_cfg.endpoint = 'HN35_Taste_M06'
    # Dyspaghia:
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/MDACC/preprocessed/dataset_full_mdacc'
    # save_dir_dataset = 'D:/MyFirstData_MDACC/dataset_mdacc_dyshagia'
    # data_prc_cfg.filename_endpoints_csv = 'dysphagia_features_MDACC.csv'
    # data_prc_cfg.endpoint = 'Dysphagia_6MO'
    patient_id_col = data_prc_cfg.patient_id_col
    logger = Logger(os.path.join(save_root_dir, data_prc_cfg.filename_data_preproc_features_logging_txt))
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_dataset)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, data_prc_cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # TODO: temporary
        # Load Excel file containing patient_id and their endpoints/labels/targets
        endpoints_csv = os.path.join(save_root_dir, data_prc_cfg.filename_endpoints_csv)
        df = pd.read_csv(endpoints_csv, sep=';')
        # Taste:
        # df = pd.read_csv(os.path.join(save_root_dir, 'features.csv'), sep=';')
        # label_mapper = {'Helemaal niet': 0, 'Een beetje': 0, 'Nogal': 1, 'Heel erg': 1}
        # df['HN35_Taste_M06'] = df['HN35_Taste_M06'].map(label_mapper)
        if len(df.columns) == 1:
            df = pd.read_csv(endpoints_csv, sep=',')

        # Copy endpoints.csv to dataset folder
        src = endpoints_csv
        dst = os.path.join(save_dir_dataset, data_prc_cfg.filename_endpoints_csv)
        copy_file(src, dst)

        # Create dictionary for the labels {label0: [patient_id1_label0, ..., patient_idm0_label0], ...,
        # labeln: [patient_id1_label1, ..., patient_idm1_label1]}
        labels_patients_dict = dict()
        # Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int, because of Excel/csv file)
        # to patient_id = '0111766' (type=str)
        # labels_patients_dict['0'] = ['%0.{}d'.format(patient_id_length) % x for x in
        #                              df[df[data_prc_cfg.endpoint] <= 2][patient_id_col].tolist()]
        # labels_patients_dict['1'] = ['%0.{}d'.format(patient_id_length) % x for x in
        #                              df[df[data_prc_cfg.endpoint] > 2][patient_id_col].tolist()]
        labels_patients_dict['0'] = ['%0.{}d'.format(patient_id_length) % x for x in
                                     df[df[data_prc_cfg.endpoint] == 0][patient_id_col].tolist()]
        labels_patients_dict['1'] = ['%0.{}d'.format(patient_id_length) % x for x in
                                     df[df[data_prc_cfg.endpoint] == 1][patient_id_col].tolist()]

        for label, patients_list in labels_patients_dict.items():
            save_dir_dataset_label = os.path.join(save_dir_dataset, str(label))
            create_folder_if_not_exists(save_dir_dataset_label)

            # Testing for a small number of patients
            if data_prc_cfg.test_patients_list is not None:
                test_patients_list = data_prc_cfg.test_patients_list
                patients_list = [x for x in test_patients_list if x in patients_list]

            for patient_id in tqdm(patients_list):
                logger.my_print('Patient_id: {id}'.format(id=patient_id))

                # TODO: currently redundant
                # Determine folder type of interest ('with_contrast'/'no_contrast')
                # TODO: temporary, for MDACC
                if use_umcg:
                    folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
                else:
                    folder_type_i = ''

                # Copy files from dataset_full to dataset/0 and dataset/1
                src = os.path.join(save_dir_dataset_full, patient_id)
                dst = os.path.join(save_dir_dataset_label, patient_id)
                copy_folder(src, dst)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_features():
    """
    Extract features: for deep learning (see load_data.py) and logistic regression model (see main.py and load_data.py).

    Note: run before: split_stratified_sampling.ipynb to generate `splits_full.csv`, then create_endpoints_csv.ipynb
        to generate `endpoints.csv`. The file `endpoints.csv` is required for this function.

    Returns:

    """
    # Initialize variables
    use_umcg = data_prc_cfg.use_umcg
    mode_list = data_prc_cfg.mode_list
    data_dir_mode_list = data_prc_cfg.save_dir_mode_list
    patient_id_length = data_prc_cfg.patient_id_length

    save_root_dir = data_prc_cfg.save_root_dir
    save_dir_dataset = data_prc_cfg.save_dir_dataset
    logger = Logger(os.path.join(save_root_dir, data_prc_cfg.filename_data_preproc_features_logging_txt))
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_dataset)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, data_prc_cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

            # Load Excel file containing patient_id, features and their endpoints/labels/targets
            df = pd.read_csv(os.path.join(save_root_dir, data_prc_cfg.filename_endpoints_csv), sep=';')

            # Select features
            cols = [data_prc_cfg.patient_id_col, data_prc_cfg.endpoint, data_prc_cfg.baseline_col] + data_prc_cfg.features + data_prc_cfg.ext_features
            df = df[cols]

            # TODO: temporary
            # save_dir_dataset = 'D:/Github/dl_ntcp_xerostomia/unfilled_holes/dataset'
            # Save features df as csv file
            for filename in [data_prc_cfg.filename_features_csv, data_prc_cfg.filename_stratified_sampling_csv]:
                df.to_csv(os.path.join(save_dir_dataset, filename), sep=';', index=False)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')

