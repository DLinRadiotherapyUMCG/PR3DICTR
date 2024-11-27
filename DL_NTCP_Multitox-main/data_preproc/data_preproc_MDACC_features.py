import pandas as pd
import numpy as np
import os
import data_preproc_config as data_prc_cfg
import data_preproc_functions

patient_id_length = 10



dose_features_MDACC =  ['Parotid_Dmean', 'OralCavity_ExtDmean',  'PCM_InfDmean', 'PCM_MedDmean', 'PCM_SupDmean', 'Submandibular_Dmean']
dose_features_MDACC = [] # ['Parotid_LDmean', 'Parotid_RDmean']





MDACC_to_UMCG_dose_col_name_mapping = {
    "Parotid_Dmean" : "Parotid_meandose",
    "OralCavity_ExtDmean" : "OralCavity_Ext_meandose",
    "PCM_InfDmean" : "PCM_Inf_meandose",
    "PCM_MedDmean" : "PCM_Med_meandose",
    "PCM_SupDmean" : "PCM_Sup_meandose",
    "Submandibular_Dmean" : "Submandibular_meandose",
}

# dose_features_MDACC = list(l_r_dose_features_MDACC.keys()) +  ['CricoDmean'	'Esophagus_CervDmean', 'OralCavity_ExtDmean',  'PCM_InfDmean', 'PCM_MedDmean', 'PCM_SupDmean', 
#                                                               'SupraglotticDmean',]


dose_features = ['Parotid_L_meandose', 'Parotid_R_meandose', 'Parotid_meandose', 'Submandibular_L_meandose', 'Submandibular_R_meandose', 'Submandibular_meandose',
                'PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'Crico_meandose', 'Supraglottic_meandose', 'OralCavity_Ext_meandose', 'BuccalMucosa_L_meandose',
                'BuccalMucosa_R_meandose', 'BuccalMucosa_meandose', 'TongueTop_meandose', 'External_meandose', 'PCM_meandose', 'Esophagus_Cerv_meandose']


#CricoDmean	Esophagus_CervDmean	OralCavity_ExtV40	OralCavity_ExtDmean	PCMDmean	PCM_InfDmean	PCM_MedDmean	PCM_SupDmean		SupraglotticDmean	BrainDmean	MandibleDmean	BrainstemDmean	Submandibular_Dmean	ExternalDmean_segmentation	External_volume_cc_segmentation



# MDACC toxicity renaming:
MDACC_tox_rename_dict = {
    "mdasi_drymouth" : "Xerostomia",
    "mdasi_choke" : "Aspiration",
    "mdasi_mucus" : "Sticky",
    "mdasi_taste" : "Taste",
    "diet_normalcy" : "Dysphagia",
}

MDACC_timeline_rename_dict = {
    "baseline" : "BSL",
    "startRT" : "W01",
    "M6" : "M06",
    "M12" : "M12",
}



def preprocess_MDACC_toxicity_features(df_mdacc):
    df_mdacc_tox = df_mdacc.copy()
    mdacc_tox_names = data_preproc_functions.get_all_combinations(list(MDACC_timeline_rename_dict.keys()), list(MDACC_tox_rename_dict.keys()), spacer="_")
    #print(mdacc_tox_names)
    df_mdacc_tox = df_mdacc_tox[[data_prc_cfg.patient_id_col] + mdacc_tox_names]

    # rename the toxicity columns (i.e. "mdasi_drymouth" -> "Xerostomia" etc.)
    for key, value in MDACC_tox_rename_dict.items():
        #df_mdacc_tox[value] = df_mdacc_tox[key]
        df_mdacc_tox.columns = df_mdacc_tox.columns.str.replace(key, value)
    # rename the dates (i.e. "baseline" -> "BSL" etc.)
    for key, value in MDACC_timeline_rename_dict.items():
        #df_mdacc_tox[value] = df_mdacc_tox[key]
        df_mdacc_tox.columns = df_mdacc_tox.columns.str.replace(key, value)

    df_mdacc_tox = swap_tox_endpoint_titles_order(df_mdacc_tox)

    return df_mdacc_tox
    


def split_dysphagia_values(value):
    if value is np.nan: # or value='nan':
        return np.nan
    try:   
        val = str(value).split(")")[0]
        val = int(val)
        return val
    except:
        return np.nan
    


def grade_range_to_label_converter(x, tox_value_mapping_dict, invert = False):
    for UMCG_label, value_range in tox_value_mapping_dict.items():
        if invert: # for dysphgia
            if x <= value_range[0] and x >= value_range[1]:
                return UMCG_label
        else:
            if x >= value_range[0] and x <= value_range[1]:
                return UMCG_label
    return np.nan

def swap_tox_endpoint_titles_order(df):
    """
    The MDACC data has the format "time_toxName", but I'm using "toxName_time" in the UMCG data. This function swaps the order of the columns to match the UMCG format.
    """
    endpoints = ["BSL", "W01", "M06", "M12"]
    
    rename_dict = {}
    for endpoint in endpoints:
        endpoint_cols = [col for col in df.columns if endpoint in col]
        for col_name in endpoint_cols:
            col_name_parts = col_name.split("_")
            new_col_name = col_name_parts[1] + "_" + col_name_parts[0]
            rename_dict[col_name] = new_col_name
    
    df.rename(columns=rename_dict, inplace=True)
    
    return df
    # df.rename(columns={'old_column_name': 'new_column_name'}, inplace=True)

def replace_missing_MDACC_BSL_values(df, toxicity_names):
    """
    
    Args:
        
    Returns: 

    """

    WO1_cols = data_preproc_functions.get_all_combinations(toxicity_names, ["W01"], spacer='_')  # W01
    BSL_cols = data_preproc_functions.get_all_combinations(toxicity_names, ["BSL"], spacer='_')  # BSL

    for tox_BSL_col, tox_W01_col in zip(BSL_cols, WO1_cols):
        df[tox_BSL_col] = df[tox_BSL_col].fillna(df[tox_W01_col])
    
    return df


def binarize_MDACC_toxicity_features(df_mdacc_tox):
    """
    Binarizes the values in the specified column of the given dataframe.
    Values equal to or above 4 are set to 1, else 0.
    """

    # thresholds for binarization
    # tox_thresholds = {
    #     "Xerostomia" : [4, '>='],
    #     "Sticky" : [4, '>='],
    #     "Taste" : [4, '>='],
    #     "Aspiration" : [5, '>='],
    #     "Dysphagia" : [50, '<='],
    # }

    # the dysphagia column is a bit different, have to split off the string part
    dysph_columns = [x for x in df_mdacc_tox.columns if "Dysphagia" in x]
    for dys_col in dysph_columns:
        df_mdacc_tox[dys_col] = df_mdacc_tox[dys_col].apply(lambda x: split_dysphagia_values(x))

        
    # put the MDACC values into the same format as UMCG (for binarising the features)
    aspiration = {"Helemaal_niet": [0,1.999],  "Een_beetje": [2,4.999],  "Nogal": [5, 7.999],  "Heel_erg": [8, 10]}
    sticky =     {"Helemaal_niet": [0,1.999],  "Een_beetje": [2,3.999],  "Nogal": [4, 6.999],  "Heel_erg": [7, 10]}
    taste =      {"Helemaal_niet": [0,1.999],  "Een_beetje": [2,3.999],  "Nogal": [4, 6.999],  "Heel_erg": [7, 10]}
    xerostomia = {"Helemaal_niet": [0,1.999],  "Een_beetje": [2,3.999],  "Nogal": [4, 6.999],  "Heel_erg": [7, 10]}

    dysphagia =  {"Grade0_1": [100,70.001],   "Grade2": [70,50.001],   "Grade3_4": [50, 0]}

    tox_thresholds = { "Aspiration" : aspiration, "Sticky" : sticky, "Taste" : taste, "Xerostomia" : xerostomia, "Dysphagia" : dysphagia}
    #tox_thresholds = {"Xerostomia" : xerostomia}


    tox_cols_to_binarise = []
    for tox_name, tox_value_mapping in tox_thresholds.items():
        tox_columns = [x for x in df_mdacc_tox.columns if tox_name in x]
        tox_cols_to_binarise += tox_columns
        invert = False if tox_name != "Dysphagia" else True
        for col in tox_columns:
            df_mdacc_tox[col] = df_mdacc_tox[col].apply(lambda x: grade_range_to_label_converter(x, tox_value_mapping, invert))
            

    df_mdacc_tox = replace_missing_MDACC_BSL_values(df_mdacc_tox, list(tox_thresholds.keys()))
    
    for tox_col_name in tox_cols_to_binarise:
        # Binarize the values in 'tox_col_name'
        df_binary_columns = pd.get_dummies(df_mdacc_tox[tox_col_name], prefix=tox_col_name)
        df_binary_columns = df_binary_columns.astype('Int64')
        
        if "Dysphagia" not in tox_col_name:
            # make a new column that joins the two highest levels of toxicity ("Nogal" and "Heel erg") together
            df_binary_columns = add_missing_tox_baseline_binary_columns(df_binary_columns, tox_col_name)
            df_binary_columns[f"{tox_col_name}_Nogal_Heel_erg"] = df_binary_columns[f"{tox_col_name}_Nogal"] + df_binary_columns[f"{tox_col_name}_Heel_erg"]
            
        # Concatenate the binarized columns with the original DataFrame
        df_mdacc_tox = pd.concat([df_mdacc_tox, df_binary_columns], axis=1)

    # binarise the original columns
    value_mapping = {"Grade0_1":int(0), "Grade2":int(1), "Grade3_4":int(1)}  ## for dysphagia
    df_mdacc_tox = df_mdacc_tox.replace(value_mapping)
    #value_mapping = {"not_at_all":0, "little":0, "moderate":1, "severe":1}
    value_mapping = {"Helemaal_niet":int(0), "Een_beetje":int(0), "Nogal":int(1), "Heel_erg":int(1)}
    df_mdacc_tox = df_mdacc_tox.replace(value_mapping)

    all_tox_cols = [col for col in df_mdacc_tox.columns if any(item in col for item in tox_thresholds.keys())]
    #all_tox_cols = [col for col in df_mdacc_tox.columns if "Xerostomia" in col or "Aspiration" in col or "Sticky" in col or "Taste" in col or "Dysphagia" in col]
    #df_mdacc_tox = df_mdacc_tox[[data_prc_cfg.patient_id_col] +all_tox_cols]
    return df_mdacc_tox




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



def preprocess_MDACC_dose_features(df_mdacc):
    """
    Preprocesses the relevant meandose columns (i.e. drop the ones we don't need)
    """
    df_mdacc_dose_features = df_mdacc[[data_prc_cfg.patient_id_col] + dose_features_MDACC]
    df_mdacc_dose_features = df_mdacc_dose_features.rename(columns=MDACC_to_UMCG_dose_col_name_mapping)
    #df_mdacc_dose_features = df_mdacc_dose_features[[data_prc_cfg.patient_id_col] + dose_features_MDACC]

    return df_mdacc_dose_features



def preprocess_MDACC_clinical_features(df_mdacc_all):
    """
    Preprocesses the MDACC features
    """
    df_mdacc_clinical_features = df_mdacc_all.copy()
    # rename some columns
    rename_dict = {
        "AGE1" : "Age",
        "Tumour location" : "Loctum2",
        "Tstage" : "T_stage",
        "Nstage" : "N_stage",
    }
    df_mdacc_clinical_features = df_mdacc_clinical_features.rename(columns=rename_dict)

    # make a Photons column
    df_mdacc_clinical_features['Photons'] = np.where((df_mdacc_clinical_features['Treatment technique'] == 'IMPT') | (df_mdacc_clinical_features['Treatment technique'] == '3-D plan'), 0, 1)
    #df                                              df_mdacc_clinical_features['Treatment technique'] == '3-D plan'], 1, 0)

    df_mdacc_clinical_features['Age'] = df_mdacc_clinical_features['Age'] / 100
    # binary encode Sex
    df_mdacc_clinical_features['Sex'] = df_mdacc_clinical_features['Sex'].replace({"Female":int(0), "Male":int(1)})

    # add CT+C column
    df_mdacc_clinical_features["CT+C"] = 0
    # same for WHO score
    df_mdacc_clinical_features["WHO"] = 0
    df_mdacc_clinical_features["Smoking"] = 0

    # only keep the features we need
    df_mdacc_clinical_features = df_mdacc_clinical_features[[data_prc_cfg.patient_id_col] + ["Age", "Sex", "Loctum2", "Photons",
                                                                                             "T_stage", "N_stage", "WHO", "Smoking",
                                                                                             "CT+C", "CT_Artefact", "P16"]]

    return df_mdacc_clinical_features
    

    
def add_missing_binarised_feature_columns(df, column_names):
    for col_name in column_names:
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

        # elif col_name == "Smoking":          # NOTE: smoking is all 0's for MDACC set
        #     smoking_mapping = {
        #         "Ja, in verleden" : int(0),   # ! Ja, inverleden becomes a 0 label !
        #         "Ja, nog steeds" : int(1),
        #         "Nee" : int(1)
        #     }
        #     df[col_name] = df[col_name].replace(smoking_mapping)
        elif col_name == "P16":
            P16_mapping = {
                None : "Niet_bepaald",
                np.nan : "Niet_bepaald",
                "Unknown" : "Niet_bepaald",
                "Niet bepaald" : "Niet_bepaald",
                "Positief" : "Positief",
                "Negatief" : "Negatief",
                "Yes" : "Positief",
                "No" : "Negatief",
                '1' : "Positief",
                '0' : "Negatief",
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
                "T4b" : "T4",
                # also catch the lower case versions
                "tx" : "Tx",
                "tis" : "Tis",
                "t0" : "T0",
                "t1" : "T1",
                "t1a" : "T1",
                "t1b" : "T1",
                "t2" : "T2",
                "t3" : "T3",
                "t4" : "T4",
                "t4a" : "T4",
                "t4b" : "T4",
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
                "N3c" : "N3",
                # also catch the lower case versions
                "nx" : "Nx",
                "n0" : "N0",
                "n1" : "N1",
                "n2" : "N2",
                "n2a" : "N2",
                "n2b" : "N2",
                "n2c" : "N2",
                "n3" : "N3",
                "n3a" : "N3",
                "n3b" : "N3",
                "n3c" : "N3",

            }
            df[col_name] = df[col_name].replace(N_stage_mapping)

        # Binarize the values in 'tox_col_name'
        df_binary_columns = pd.get_dummies(df[col_name], prefix=col_name)
        df_binary_columns = df_binary_columns.astype('Int64')

        #loctum_mapping
        #print(col_name)
        if "Loctum2" in col_name:
            binary_col_names = data_preproc_functions.get_all_combinations([col_name], list(set(loctum_mapping.values())), spacer="_")
            #print(binary_col_names)
            df_binary_columns = add_missing_binarised_feature_columns(df_binary_columns, binary_col_names)
        elif col_name == "WHO":
            binary_col_names = data_preproc_functions.get_all_combinations([col_name], list(set(WHO_mapping.values())), spacer="_")
            #print(binary_col_names)
            df_binary_columns = add_missing_binarised_feature_columns(df_binary_columns, binary_col_names)
        elif col_name == "T_stage":
            binary_col_names = data_preproc_functions.get_all_combinations([col_name], list(set(T_stage_mapping.values())), spacer="_")
            #print(binary_col_names)
            df_binary_columns = add_missing_binarised_feature_columns(df_binary_columns, binary_col_names)
        elif col_name == "N_stage":
            binary_col_names = data_preproc_functions.get_all_combinations([col_name], list(set(N_stage_mapping.values())), spacer="_")
            #print(binary_col_names)
            df_binary_columns = add_missing_binarised_feature_columns(df_binary_columns, binary_col_names)
        elif col_name == "P16":
            binary_col_names = data_preproc_functions.get_all_combinations([col_name], list(set(P16_mapping.values())), spacer="_")
            #print(binary_col_names)
            df_binary_columns = add_missing_binarised_feature_columns(df_binary_columns, binary_col_names)
        

        # make the T-stage and N-stage column names shorter
        df_binary_columns.columns = df_binary_columns.columns.str.replace('T_stage_', '')
        df_binary_columns.columns = df_binary_columns.columns.str.replace('N_stage_', '')

        #

        # Concatenate the binarized columns with the original DataFrame
        df = pd.concat([df, df_binary_columns], axis=1)
    
    # add missing features

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
    #print(specific_cols)
    if specific_cols == None:
        df_filtered = df.dropna()
    else:
        df_filtered = df.dropna(subset=specific_cols)  # drop all patients with missing values
    num_dropped = len(df) - len(df_filtered)
    print(f'Dropped {num_dropped} patients because of missing values in the following columns: {cols_with_missing_values}')

    return df_filtered




def load_MDACC_features_excel():
    """
    The function loads the MDACC features from the excel file. The file already contains the features (IDs, clinical features, baselines, endpoints, etc.)
    as well as the dose parameters (that are used by the CITOR models).
    """

    df_mdacc_all = pd.read_excel("//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Daniel MacRae/MDACC_database-clean_Daniel.xlsx")
    df_mdacc_all = df_mdacc_all.rename(columns={"newID": data_prc_cfg.patient_id_col})

    df_mdacc_all[data_prc_cfg.patient_id_col] = [str(item).zfill(patient_id_length) for item in df_mdacc_all[data_prc_cfg.patient_id_col]]

    df_mdacc_dose_features = preprocess_MDACC_dose_features(df_mdacc_all)

    df_mdacc_clinical_features = preprocess_MDACC_clinical_features(df_mdacc_all)
    #print("DANIEL")
    #print(df_mdacc_clinical_features.columns)
    

    df_mdacc_toxicity_columns = preprocess_MDACC_toxicity_features(df_mdacc_all)
    
    # merge the features dataframes
    df_merged = pd.merge(df_mdacc_clinical_features, df_mdacc_dose_features, on=data_prc_cfg.patient_id_col, how='inner')
    #df_merged = pd.merge(df_merged, df_mdacc_dose_features, on=data_prc_cfg.patient_id_col, how='inner')
    df_merged = pd.merge(df_merged, df_mdacc_toxicity_columns, on=data_prc_cfg.patient_id_col, how='inner')
    #print(len(df_merged))
    #print(df_merged.columns)

    df_merged["Split"] = "test"


    df_merged = binarise_feature_columns(df_merged, ["Loctum2", "T_stage", "N_stage", "WHO", "Smoking", "P16"])
    df_merged = binarize_MDACC_toxicity_features(df_merged)
    #print(df_merged.columns)


    df_merged.sort_values(by=data_prc_cfg.patient_id_col, inplace=True)
    # save the dataframe
    df_merged.to_csv("MDACC_features_all.csv", index=False, sep=';')
    df_merged.to_excel("MDACC_features_all.xlsx", index=False)

    # MAKE ANOTHER DATAFRAME THAT ONLY CONTAINS PATIENTS THAT HAVE ALL FEATURES PRESENT

    # make a function to drop patients without any endpoints, and that are missing at least one input
    main_columns = [data_prc_cfg.patient_id_col, "Sex", "Age", "CT+C", "CT_Artefact", "Loctum2", "Photons", "T_stage", "N_stage", "Smoking", "P16", "WHO"]  #["PatientID", "Sex", "Age", "CT+C_available", "CT_Artefact", "Loctum2", "Photons", "Smoking", "T_stage", "N_stage", "P16", "WHO"]

 
    target_toxicities = ["Xerostomia", "Aspiration", "Sticky", "Taste", "Dysphagia"]   
    baseline_timepoints = ["BSL"]
    endpoint_timepoints = ["M06"]
    toxicity_baseline_features = data_preproc_functions.get_all_combinations(target_toxicities, baseline_timepoints, spacer="_")
    toxicity_endpoint_features = data_preproc_functions.get_all_combinations(target_toxicities, endpoint_timepoints, spacer="_")
   
    print(len(df_merged))
    df_shortened = remove_patients_missing_endpoints(df_merged, toxicity_endpoint_features)
    print(len(df_shortened))

    all_input_features = main_columns + toxicity_baseline_features #+ toxicity_endpoint_features
    
    print(toxicity_baseline_features)
    df_shortened = keep_patients_with_all_data_present(df_shortened, all_input_features)
    print(len(df_shortened))
    

    df_shortened.sort_values(by=data_prc_cfg.patient_id_col, inplace=True)
    # fill missing endpoint columns with a -1
    df_shortened[toxicity_endpoint_features] = df_shortened[toxicity_endpoint_features].fillna(-1)

    # save the dataframe
    df_shortened.to_csv("MDACC_features_present_M06.csv", index=False, sep=';')
    df_shortened.to_excel("MDACC_features_present_M06.xlsx", index=False)

    per_tox_subfolder = "per_toxicity_csvs"
    os.makedirs(per_tox_subfolder, exist_ok=True)
    # Iterate over each column name in the list
    
    for column_name in toxicity_endpoint_features:
        # Drop rows with -1 in the current column
        df_copy = df_shortened[df_shortened[column_name] != -1].copy()

        print(column_name,"has",len(df_copy), "patients.")

        # Save the resulting dataframe
        df_copy.to_csv(os.path.join(per_tox_subfolder, f'{column_name}.csv'), index=False, sep=";")

# asp 310
# dysph 348
# sticky 310
# taste 308
# xerost 310

def remove_patients_missing_endpoints(df, toxicity_endpoint_features):
    """
    Removes the patients that do not have at least 1 endpoint. 
    Args:
        df: a dataframe.
    Returns: 
        df: the same dataframe, but only with patients that have at least 1 endpoint
    """
    for col in toxicity_endpoint_features:
        num_missing = len(df) - len(df.dropna(subset=[col]))
        if num_missing > 0:
            print(f"    {num_missing} patients have a missing value in column {col}")
            #cols_with_missing_values.append(cols_with_missing_values)
    df = df.dropna(subset=toxicity_endpoint_features, how='all')

    return df





def main():
    #print(dose_features_MDACC)
    load_MDACC_features_excel()


if __name__ == '__main__':
    main()
    
