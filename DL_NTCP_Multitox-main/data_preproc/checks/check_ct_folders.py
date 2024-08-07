import os
import time
import json
import pandas as pd
import pydicom as pdcm
from glob import glob
from tqdm import tqdm
from pydicom import dcmread
import data_preproc_config as data_prc_cfg



def get_metadata_rtdose(path_rtdose_file):
    # Load RTDOSE file
    ds = dcmread(path_rtdose_file)

    # Construct metadata_rtdose
    metadata_rtdose = json.loads(ds.to_json())

    return metadata_rtdose


def get_metadata_ct(path_ct):
    # Load CT slice
    slices_path = glob(path_ct + '/*')
    # slices_path = sort_human(slices_path)
    slice = pdcm.read_file(slices_path[0])

    # Construct metadata_ct
    metadata_ct = json.loads(slice.to_json())

    return metadata_ct


def get_metadata_rtstruct(path_rtstruct_file):
    # Load RTSTRUCT file
    ds = dcmread(path_rtstruct_file)

    # Construct metadata_rtdose
    metadata_rtstruct = json.loads(ds.to_json())

    return metadata_rtstruct

def get_reference_frame_ct(path_ct):
    metadata = get_metadata_ct(path_ct)
    frame_of_ref_uid = metadata[data_prc_cfg.frame_of_ref_uid_tag]['Value'][0]

    return frame_of_ref_uid

def get_reference_frame_rtdose(path_rtdose_file):
    metadata = get_metadata_rtdose(path_rtdose_file)
    frame_of_ref_uid = metadata[data_prc_cfg.frame_of_ref_uid_tag]['Value'][0]

    return frame_of_ref_uid
    
def get_reference_frame_rtstruct(path_rtstruct_file):
    metadata = get_metadata_rtstruct(path_rtstruct_file)
    frame_of_ref_uid = metadata['30060010']['Value'][0][data_prc_cfg.frame_of_ref_uid_tag]['Value'][0]

    return frame_of_ref_uid



def check_if_CT_matches_an_RTDOSE(patient_folder, patient_folder_types, CT_UID):
    #RTDOSE_UIDS = []
    matching_RTDOSE_UID = False
    for f_t in patient_folder_types:
        patient_foldertype_dir = os.path.join(patient_folder, f_t)
        RTSTRUCT_folder_dirs = glob(patient_foldertype_dir + '*/RTDOSE/*') + glob(patient_foldertype_dir + '/*/RTDOSE/*')
        #print("   ", RTSTRUCT_folder_dirs)

        for path in RTSTRUCT_folder_dirs:
            ref_uid = get_reference_frame_rtdose(path)

            if ref_uid == CT_UID:
                matching_RTDOSE_UID = True
                return matching_RTDOSE_UID
    return matching_RTDOSE_UID

def get_all_RTSTRUCT_UIDs(patient_folder, patient_folder_types):
    RTSTRUCT_UIDS = []
    for f_t in patient_folder_types:
        patient_foldertype_dir = os.path.join(patient_folder, f_t)
        RTSTRUCT_folder_dirs = glob(patient_foldertype_dir + '*/RTSTRUCT/*') + glob(patient_foldertype_dir + '/*/RTSTRUCT/*')
        # for root, dirs, files in os.walk(patient_folder):
        #     if "RTSTRUCT" in dirs:
        #         rtstruct_dir = os.path.join(root, "RTSTRUCT")
        for path in RTSTRUCT_folder_dirs:
            #file_path = os.path.join(rtstruct_dir, file)
            print(path)
            ref_uid = get_reference_frame_rtstruct(path)
            RTSTRUCT_UIDS.append(ref_uid)
            print(ref_uid)
    return RTSTRUCT_UIDS



def check_if_item_in_list(item, list_of_items):
    return any(item == i for i in list_of_items)



def check_if_CT_UID_matches_available_RTDOSE_images(patient_folder_dir, ct_scan_folder_type, patient_folder_types):
    """
    A function that checks if the CT image in the `ct_scan_folder_type` folder has a corresponding RTDOSE image (which should have the same reference frame UID as the CT image). If there is a corresponding RTDOSE and RTSTRUCT image, the function returns True, otherwise it returns False.
    Args:
        patient_folder_dir (str): the path to the patient folder
        ct_scan_folder_type (str): the folder type that contains the CT images (e.g. "with_contrast" or "no_contrast")
        patient_folder_types (list): a list of folder types that the patient has (e.g. ["with_contrast", "no_contrast"])
    Returns:
        bool: True if there is a RTDOSE with the same UID (reference frame), otherwise False
    """

    # get the path to the folder that contains the CT images for this `folder_type` (e.g. "with_contrast" or "no_contrast")
    patient_folder_type_dir = os.path.join(patient_folder_dir, ct_scan_folder_type)
    ct_files_dir = glob(patient_folder_type_dir + '*\\CT') + glob(patient_folder_type_dir + '\\*\\CT')
    
    ct_files_dir = ct_files_dir[0] # there should only be one CT folder ! 
    #print(ct_files_dir)
    
    # get the reference frame UID of this CT image
    CT_UID = get_reference_frame_ct(ct_files_dir)

    # now see if there is a corresponding RTDOSE and RTSTRUCT image with the same reference frame UID
    # RTDOSE first
    has_matching_RTDOSE = check_if_CT_matches_an_RTDOSE(patient_folder_dir, patient_folder_types, CT_UID)
    #print("matching RTDOSE", has_matching_RTDOSE)

    # get the path to the folder that contains the RTDOSE images for this `folder_type` (e.g. "with_contrast" or "no_contrast")
    #RTSTRUCT_UIDs = get_all_RTSTRUCT_UIDs(patient_folder_dir, patient_folder_types)
    return has_matching_RTDOSE
    