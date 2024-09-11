import csv
import os
from typing import Optional, List

import numpy as np
import pandas as pd
from monai.transforms import ScaleIntensityRange, MapLabelValue, Compose, RandSpatialCrop, RandFlip, RandAffine, \
    RandRotate, Rand3DElastic
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.utils.center_crop_3d import center_crop_3d
from src.utils.data_equalizer import data_split, get_delimiter

def preprocess_image_stack(image_stack, config):
    """
    Preprocesses a single image.
    :param image_stack:
    :param config:
    :return:
    """
    if(config['data']['preprocessing']['isEnabled']):
        # Standardization
        ct_scaler = ScaleIntensityRange(a_min=config['data']['preprocessing']['ct']['a_min'],
                                        a_max=config['data']['preprocessing']['ct']['a_max'],
                                        b_min=config['data']['preprocessing']['ct']['b_min'],
                                        b_max=config['data']['preprocessing']['ct']['b_max'],
                                        clip=config['data']['preprocessing']['ct']['clip'],
                                        dtype=np.float32)
        rtdose_scaler = ScaleIntensityRange(a_min=config['data']['preprocessing']['rtdose']['a_min'],
                                            a_max=config['data']['preprocessing']['rtdose']['a_max'],
                                            b_min=config['data']['preprocessing']['rtdose']['b_min'],
                                            b_max=config['data']['preprocessing']['rtdose']['b_max'],
                                            clip=config['data']['preprocessing']['rtdose']['clip'],
                                            dtype=np.float32)
        segmentation_map_transform = MapLabelValue(
            orig_labels=[index for index, x in enumerate(config['data']['preprocessing']['segmentation_map']['target_labels'])],
            target_labels=config['data']['preprocessing']['segmentation_map']['target_labels'])

        image_stack[0] = ct_scaler(image_stack[0])
        image_stack[1] = rtdose_scaler(image_stack[1])
        image_stack[2] = segmentation_map_transform(image_stack[2])

    return image_stack

def augmentation(config):
    transforms = []

    if(config['data']['augmentation']['isEnabled']):
        augmentation = config['data']['augmentation']['augmentation_list']
        
        # Define the MONAI transformations
        prob = config['data']['augmentation']['prob']
        strength = config['data']['augmentation']['strength']
                
        possible_augmentation = ['crop','flip','affine','rotate']
        
        if 'crop' in augmentation:
            transforms.append(RandSpatialCrop(roi_size=config['data']['preprocessing']['crop_shape'], random_size=False, random_center=True))
        
        if 'flip' in augmentation:
            transforms.append(RandFlip(prob=prob, spatial_axis=1))
        
        if 'affine' in augmentation:
            transforms.append(RandAffine(prob=prob, translate_range=(7 * strength,) * 3, scale_range=(0.07 * strength,) * 3, padding_mode='border', mode='bilinear'))
        
        if 'rotate' in augmentation:
            transforms.append(RandRotate(prob=prob, range_x=(3.14 / 24 * strength), align_corners=True, padding_mode='border', mode='bilinear'))
        
        for aug in augmentation:
            if aug not in possible_augmentation:
                print("Augmentation: " + aug + " is not implimented and therefor not applied")
    
    
    return Compose(transforms)


class HNCDataset(Dataset):
    """
    Dataset class for the HNC dataset.
    Returns a tuple of image_stack, clinical_features, and label.
    """

    def __init__(self, csv_path, config, patient_ids, augment = False, split = False, train = True, splitVar = "Split"):
        self.images_path, self.label_column = config['paths']['images'], config['columns']['label']
        self.config = config
        self.augment = augment

        # Read the csv file
        delimiterFound = get_delimiter(csv_path)
        df = pd.read_csv(csv_path, delimiter=delimiterFound, dtype={'PatientID': str})


        # if(train):
        #     df = df[df[splitVar] == "train"]
        # else:
        #     df = df[df[splitVar] == "val"]     

        # Filter the data based on the patient ids, if provided.
        if patient_ids:
            df = df[df['PatientID'].isin(patient_ids)]

        self.df = df

        # Define the MONAI transformations
        prob = config['augmentation']['prob']
        strength = config['augmentation']['strength']

        self.transforms = augmentation(config)

        for transform in self.transforms.transforms:
            transform.set_random_state(seed=config['general']['seed'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        patient_id = self.df.iloc[idx]['PatientID']
        # Construct the file paths for the CT, dose, and segmentation map images
        ct_path = self.images_path + str(patient_id).rjust(7,'0') + '/ct.npy'
        dose_path = self.images_path + str(patient_id).rjust(7,'0') + '/rtdose.npy'
        segmentation_map_path = self.images_path + str(patient_id).rjust(7,'0') + '/segmentation_map.npy'

        # Load the images
        ct = np.load(ct_path)
        dose = np.load(dose_path)
        segmentation_map = np.load(segmentation_map_path)

        # Need to be 4 dimensions
        if(len(ct.shape) == 3):
            ct = np.expand_dims(ct,axis = 0)
            dose = np.expand_dims(dose,axis = 0)
            segmentation_map = np.expand_dims(segmentation_map,axis = 0)

        # Stack the images vertically
        image_stack = np.vstack((ct, dose, segmentation_map), dtype=np.float32)

        # Preprocess the image_stack
        image_stack = preprocess_image_stack(image_stack, self.config).astype(np.float32)

        # Apply the MONAI transformations
        if self.augment:
            image_stack = self.transforms(image_stack)

        # If cropping is enabled, crop the image
        if self.config['data']['preprocessing']['crop']:
            cropped_image_stack = np.empty((image_stack.shape[0], *self.config['data']['preprocessing']['crop_shape']))
            for i in range(image_stack.shape[0]):
                cropped_image_stack[i] = center_crop_3d(image_stack[i], self.config['data']['preprocessing']['crop_shape'])
            image_stack = cropped_image_stack.astype(np.float32)

        # Get the label
        label = np.array([self.df[self.df['PatientID'] == patient_id][self.label_column].values[0]]).astype(np.float32)

        # Get the clinical features
        clinical_features = [self.df[self.df['PatientID'] == patient_id][feature].values[0] for feature in
                             self.config['columns']['clinical_features']]
        clinical_features = np.array(clinical_features).astype(np.float32)

        return image_stack, clinical_features, label
    

class ToxDataset(Dataset):
    """
    Dataset class for the HNC dataset.
    Returns a tuple of image_stack, clinical_features, and label.
    """

    def __init__(self, config, dataframe):
        self.images_path, self.label_column = config['paths']['images'], config['columns']['label']
        self.config = config

        self.df = dataframe

        self.transforms = augmentation(config)

        for transform in self.transforms.transforms:
            transform.set_random_state(seed=config['general']['seed'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Check if the dataframe is not none
        if(self.df.shape[0] == 0):
            print("No data available in dataset.")
            return None
        
        # Get selected patient
        patient_id = self.df.iloc[idx]['PatientID']

        pathPatientDir = self.images_path + str(patient_id).rjust(7,'0')
        if(os.path.exists(pathPatientDir) == False):
            pathPatientDir = self.images_path + str(patient_id)

        # Construct the file paths for the CT, dose, and segmentation map images
        ct_path = pathPatientDir + '/ct.npy'
        dose_path = pathPatientDir + '/rtdose.npy'
        segmentation_map_path = pathPatientDir + '/segmentation_map.npy'

        # Load the images
        ct = np.load(ct_path)
        dose = np.load(dose_path)
        segmentation_map = np.load(segmentation_map_path)

        # Need to be 4 dimensions
        if(len(ct.shape) == 3):
            ct = np.expand_dims(ct,axis = 0)
            dose = np.expand_dims(dose,axis = 0)
            segmentation_map = np.expand_dims(segmentation_map,axis = 0)

        # Stack the images vertically
        image_stack = np.vstack((ct, dose, segmentation_map), dtype=np.float32)

        # Preprocess the image_stack
        image_stack = preprocess_image_stack(image_stack, self.config).astype(np.float32)

        # Apply the MONAI transformations
        if self.config['data']['augmentation']['isEnabled']:
            image_stack = self.transforms(image_stack)

        # If cropping is enabled, crop the image
        if self.config['data']['preprocessing']['crop']:
            #print("Cropping")
            cropped_image_stack = np.empty((image_stack.shape[0], *self.config['data']['preprocessing']['crop_shape']))
            for i in range(image_stack.shape[0]):
                cropped_image_stack[i] = center_crop_3d(image_stack[i], self.config['data']['preprocessing']['crop_shape'])
            image_stack = cropped_image_stack.astype(np.float32)

        # Get the label
        label = np.array([self.df[self.df['PatientID'] == patient_id][self.label_column].values[0]]).astype(np.float32)

        # Get the clinical features
        clinical_features = [self.df[self.df['PatientID'] == patient_id][feature].values[0] for feature in
                             self.config['columns']['clinical_features']]
        clinical_features = np.array(clinical_features).astype(np.float32)

        return image_stack, clinical_features, label
