import os
import misc
import pandas as pd

use_umcg = True
"""
SET FILEPATHS
"""

#### INPUTS ####

processed_images_dataset_path = '//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Daniel MacRae/DL_NTCP_Multitox/datasets/dataset_dan/patients'    ## This path is where the preproccessed image files (i.e. 100x100x100 npy arrays) are saved
toxicity_endpoints_file_path = "//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Daniel MacRae/CITOR_REDCAP_clinical_data_important_variables_combined.xlsx"
dose_info_file_path = "//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Daniel MacRae/extracted_dose_16_5_2022.csv" ## the path to a file where the mean doses can be processed
exclude_list_path = "H:/preprocess_multitox/exclude_patients_umcg.csv"

image_keys = ['ct', 'rtdose', 'segmentation_map']
if use_umcg:
   patient_id_length = 7
else:
   patient_id_length = 10

#### OUTPUTS ####
save_dataset_folder = "H:/multitox_dataset"    ## this is the folder that all of the results will be put into
logging_folder = os.path.join(save_dataset_folder, "logs")  ## a subfolder of that, where some logs can be saved


logger_txt_output_path = os.path.join(save_dataset_folder, "log.txt")
missing_structs_save_path = os.path.join(logging_folder, "Patients with Missing Structures.xlsx")





## Make the output folders (if they don't already exist)
## NOTE: maybe consider checking if the folder already exists, and if it should warn that it does (changing it might not be desired)
os.makedirs(save_dataset_folder, exist_ok=True)
os.makedirs(logging_folder, exist_ok=True)

"""
SET IMPORTANT PARAMETERS
"""

## (Clinical) columns to keep
main_columns = ["PatientID", "Sex", "Age", "CT+C_available", "CT_Artefact", "Loctum2", "Photons"] 

# which toxicities we're modelling  
target_toxicities = ["Xerostomia", "Aspiration", "Sticky", "Taste", "Dysphagia"]   
                        # ["Xerostomia", "Aspiration", "Sticky", "Taste", "Dysphagia"]
# timepoints we want to use
baseline_timepoints = ["W01"]  # which baseline to use ["BSL", "W01"]
target_timepoints = ["M06"]    # what the endpoint time we're trying to predict is
                               # ["M06", "M12", "M18"]    
                                # can do: ["M06", "M12", "M18", "M24"]  but theres no dysphgia column for M24... so you'll get an error :/

## automatically generate the column names: (e.g. "Xerostomia_W01" or "Aspiration_M06")
toxicity_baseline_columns = misc.get_all_combinations(target_toxicities, baseline_timepoints, spacer="_")
toxicity_endpoint_columns = misc.get_all_combinations(target_toxicities, target_timepoints, spacer="_")

## which word to use when parsing the `dose_info_file_path` file to get the 
dose_features = ['Parotid_L_meandose', 'Parotid_R_meandose', 'Parotid_meandose', 'Submandibular_L_meandose', 'Submandibular_R_meandose', 'Submandibular_meandose',
                'PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'Crico_meandose', 'Supraglottic_meandose', 'OralCavity_Ext_meandose', 'BuccalMucosa_L_meandose',
                'BuccalMucosa_R_meandose', 'BuccalMucosa_meandose', 'TongueTop_meandose', 'External_meandose', 'PCM_meandose', 'Esophagus_Cerv_meandose']
                # None
meandose_target_word = 'meandose'  # if you want to just include everything that has this word in it
                                    # only gets used if 'dose_features' is not None
# helps automatically retrieve: 
# ['Parotid_L_meandose', 'Parotid_R_meandose', 'Parotid_meandose', 'Submandibular_L_meandose', 'Submandibular_R_meandose', 'Submandibular_meandose',
# 'PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'Crico_meandose', 'Supraglottic_meandose', 'OralCavity_Ext_meandose', 'BuccalMucosa_L_meandose',
# 'BuccalMucosa_R_meandose', 'BuccalMucosa_meandose', 'TongueTop_meandose', 'External_meandose', 'PCM_meandose', 'Esophagus_Cerv_meandose']


## Patients to exclude (a manual list):
exclude_list_manual = ["3573430", "5610140"]
# also include the csv of excluded patients
exclude_list_csv_patient_IDs = [str(item).zfill(patient_id_length) for item in pd.read_csv(exclude_list_path, delimiter=';', index_col=0).index]
# combine the two lists
exclude_list = exclude_list_manual + exclude_list_csv_patient_IDs


"""
Making test set
"""

test_set_size = 0.2
#validation_set_size = 0.2

## columns to perform the stratified sampling on
stratify_endpoints = True
other_stratify_columns = ["CT+C_available", "CT_Artefact", "Loctum2", "Photons"]
strata_groups = other_stratify_columns + stratify_endpoints*toxicity_endpoint_columns   # make the stratification groups


"""
INFO
"""

## Names of stuctures (for the segmentation maps)
parotis_structures = ['parotis_li', 'parotis_re']
submand_structures = ['submandibularis_li', 'submandibularis_re']
lower_structures = ['crico', 'thyroid']
mandible_structures = ['mandible']
other_structures = ['glotticarea', 'oralcavity_ext', 'supraglottic', 'buccalmucosa_li',
                    'buccalmucosa_re', 'pcm_inf', 'pcm_med', 'pcm_sup', 'esophagus_cerv']
structures_compound_list = [parotis_structures, submand_structures, lower_structures, mandible_structures,
                            other_structures]
structures_uncompound_list = [i for x in structures_compound_list for i in x]
# E.g. structures_uncompound_list: ['parotid_li', 'parotid_re', 'submandibular_li', ...]

structures_names_to_values = dict(zip(structures_uncompound_list,                            ## organ names values are the keys, segmap values are the dictionary values 
                             range(1, len(structures_uncompound_list) + 1)))                ## e.g. {"parotis_li" : 1}

structures_values_to_names = dict(zip(range(1, len(structures_uncompound_list) + 1),         ## segmap values are the keys, names are the values
                             structures_uncompound_list))                                   ## e.g. {1 : "parotis_li"}
