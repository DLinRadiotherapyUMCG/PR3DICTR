"""
Determine which folder type (e.g. 'with_contrast' and/or 'no_contrast') should be considered:
    - For each patient_id:
        - If 'with_contrast' is available: select 'with_contrast'.
        - Else: select 'no_contrast'.
"""
import os
import time
import pandas as pd
from tqdm import tqdm

import data_preproc_config as data_prc_cfg
from data_preproc_functions import Logger, sort_human
from checks.check_ct_folders import check_if_CT_UID_matches_available_RTDOSE_images

def main():
    # Initialize variables
    mode_list = data_prc_cfg.mode_list
    data_dir_mode_list = data_prc_cfg.save_dir_mode_list
    patient_id_length = data_prc_cfg.patient_id_length

    save_root_dir = data_prc_cfg.save_root_dir
    patient_folder_types = data_prc_cfg.patient_folder_types
    logger = Logger(output_filename=None)
    start = time.time()

    logger.my_print('Patient folder types (highest priority first): {}'.format(patient_folder_types))
    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = data_prc_cfg.data_dir.format(d_i)

        # Get all patient_ids
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)
        # Remove items with underscore from the list # DANIEL: due to new folders with underscore added in February 2024
        patients_list = [x for x in patients_list if '_' not in x]
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # TODO: temporary: consider all patients, for exclude_patients.csv
        # Testing for a small number of patients
        if data_prc_cfg.test_patients_list is not None:
            test_patients_list = data_prc_cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        df = pd.DataFrame(columns=['Patient_id', 'Folder', 'Comments'])
        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {}'.format(patient_id))

            # Get all folders
            patient_folder = os.path.join(data_dir, patient_id)
            all_folders_i = os.listdir(patient_folder)

            for f_t in patient_folder_types:
                if f_t in all_folders_i:
                    #print(patient_id, "HAS", f_t)
                    
                    has_matching_RTDOSE = check_if_CT_UID_matches_available_RTDOSE_images(patient_folder, f_t, patient_folder_types)
                    if has_matching_RTDOSE:
                        folder_type_i = f_t
                        break
                else:
                    folder_type_i = None

            if len(all_folders_i) > 1:
                logger.my_print('\tAll folders: {}'.format(all_folders_i))
                logger.my_print('\tUsing folder type: {}'.format(folder_type_i))

            df_i = pd.DataFrame({'Patient_id': patient_id, 'Folder': folder_type_i}, index=[0])
            df = pd.concat([df, df_i], axis=0)

        # Save to csv
        df = df.sort_values(by=['Patient_id', 'Folder'])
        df.to_csv(os.path.join(save_root_dir, data_prc_cfg.filename_patient_folder_types_csv), sep=';', index=False)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')
    logger.close()
    del logger


if __name__ == '__main__':
    main()




