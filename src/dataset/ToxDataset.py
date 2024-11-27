import csv
import os
import random
from typing import Optional, List

import numpy as np
import pandas as pd
from monai.transforms import ScaleIntensityRange, MapLabelValue, Compose, RandSpatialCrop, RandFlip, RandAffine, \
    RandRotate, Rand3DElastic, RandGaussianNoise
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

from src.utils.center_crop_3d import center_crop_3d
from src.utils.data_equalizer import data_split, get_delimiter

def preprocess_image_stack(image_stack, name_stack, config):
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

        for i in range(len(name_stack)):
            name = name_stack[i].lower()
            if("ct" in name):
                image_stack[i] = ct_scaler(image_stack[i])
            elif("dose" in name):
                image_stack[i] = rtdose_scaler(image_stack[i])
            elif("mask" in name):
                image_stack[i] = segmentation_map_transform(image_stack[i])
            elif("mri" in name):
                #TODO: preprocessing mri T1 and T2?
                pass
            elif("pet" in name):
                #TODO: Preprocessing pet data
                pass
            else:
                raise Exception(f"Error: Could not find the input image type base on given name: '{name}'. Please possible options or change naming of file.")

    return image_stack

def check_augmentation_Isenabled(config, augName):
    keysFound = config['data']['augmentation']['list'].keys()
    if(augName in keysFound):
        return config['data']['augmentation']['list'][augName]['Isenabled']
    return False


def augmentation(config):
    transforms = []

    if(config['data']['augmentation']['isEnabled']):
        augmentationsFound = config['data']['augmentation']['list'].keys()
        
        # Define the MONAI transformations
        #prob = config['data']['augmentation']['prob']
        #strength = config['data']['augmentation']['strength']
                
        possible_augmentation = ['crop','flip','affine','rotate',"noise"]
        
        if check_augmentation_Isenabled(config,'crop'):#'crop' in augmentation:
            transforms.append(RandSpatialCrop(roi_size=config['data']['preprocessing']['crop_shape'], random_size=False, random_center=True))
        
        #if 'translate' in augmentation:
            #transforms.append(RandSpatialCrop(roi_size=config['data']['preprocessing']['crop_shape'], random_size=False, random_center=True))

        if check_augmentation_Isenabled(config,'flip'): #'flip' in augmentation:
            prob = config['data']['augmentation']['list']['flip']['prob']
            transforms.append(RandFlip(prob=prob, spatial_axis=2))
        
        if check_augmentation_Isenabled(config,'affine'):# 'affine' in augmentation:
            prob = config['data']['augmentation']['list']['affine']['prob']
            tr_max = np.random.randint(0, config['data']['augmentation']['list']['affine']['translate_max'], 3)
            sc_max = np.random.random(3)*config['data']['augmentation']['list']['affine']['scale_max']
            sc_max[2] = sc_max[2]+config['data']['augmentation']['list']['affine']['z_scale']
            transforms.append(RandAffine(prob=prob, translate_range=tr_max, scale_range=sc_max, padding_mode='border', mode='bilinear'))
        
        if check_augmentation_Isenabled(config,'rotate'): #'rotate' in augmentation:
            prob = config['data']['augmentation']['list']['rotate']['prob']
            rot_max = config['data']['augmentation']['list']['rotate']['max']/180*3.14 # turn to rad
            transforms.append(RandRotate(prob=prob, range_x=(rot_max), align_corners=True, padding_mode='border', mode='bilinear'))
        
        if check_augmentation_Isenabled(config, 'noise'):
            prob = config['data']['augmentation']['list']['noise']['prob']
            noise_mean = config['data']['augmentation']['list']['noise']['mean']
            noise_std = config['data']['augmentation']['list']['noise']['std']
            transforms.append(RandGaussianNoise(prob = prob, mean = noise_mean, std = noise_std))

        for aug in augmentationsFound:
            if aug not in possible_augmentation:
                print("Augmentation: " + aug + " is not implimented and therefor not applied")
    
    return Compose(transforms)

class ToxDataset(Dataset):
    """
    Dataset class
    Returns a tuple of image_stack, clinical_features, and label.
    """

    def __init__(self, config, dataframe):
        self.images_path, self.label_columns = config['paths']['images'], config['columns']['label']
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

        # Construct the file paths for the input image channels
        imageInputs = self.config["data"]["imageKeys"]
        np_image_array = []
        for i in range(len(imageInputs)):
            imagepath = os.path.join(pathPatientDir, imageInputs[i] + '.npy')
            np_image_array.append(np.load(imagepath))

        # Need to be 4 dimensions
        if(len(np_image_array[0].shape) == 3):
            for i in range(len(np_image_array)):
                np_image_array[i] = np.expand_dims(np_image_array[i],axis=0)

        # Stack the images vertically
        image_stack = np.vstack(np_image_array, dtype=np.float32)

        # Preprocess the image_stack
        image_stack = preprocess_image_stack(image_stack, imageInputs, self.config).astype(np.float32)

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
        labels = np.array([self.df[self.df['PatientID'] == patient_id][self.label_columns].values[0]]).astype(np.float32)

        # Get the clinical features
        clinical_features = [self.df[self.df['PatientID'] == patient_id][feature].values[0] for feature in
                             self.config['columns']['clinical_features']]
        clinical_features = np.array(clinical_features).astype(np.float32)

        return image_stack, clinical_features, labels    ### DANIEL ATTENTION MAP: also return patient_id !!!


class ToxMonaiDataset(Dataset):
    """
    Dataset class for the HNC dataset.
    Returns a tuple of image_stack, clinical_features, and label.
    """

    def __init__(self, config, dataframe):
        self.images_path, self.label_columns = config['paths']['images'], config['columns']['label']
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

        # Construct the file paths for the input image channels
        imageInputs = self.config["data"]["imageKeys"]
        np_image_array = []
        for i in range(len(imageInputs)):
            imagepath = os.path.join(pathPatientDir, imageInputs[i] + '.npy')
            np_image_array.append(np.load(imagepath))

        # Need to be 4 dimensions
        if(len(np_image_array[0].shape) == 3):
            for i in range(len(np_image_array)):
                np_image_array[i] = np.expand_dims(np_image_array[i],axis=0)

        # Stack the images vertically
        image_stack = np.vstack(np_image_array, dtype=np.float32)

        # Preprocess the image_stack
        image_stack = preprocess_image_stack(image_stack, imageInputs, self.config).astype(np.float32)

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
        labels = np.array([self.df[self.df['PatientID'] == patient_id][self.label_columns].values[0]]).astype(np.float32)


        #labels_dict = {label_col : torch.tensor(self.df[self.df['PatientID'] == patient_id][label_col].values[0]) for label_col in self.label_columns}

        # print(self.df[self.df['PatientID'] == patient_id][self.label_columns].values[0])
        # print(self.label_columns)
        # print(labels_dict)
        # print(patient_id)

        # Get the clinical features
        clinical_features = [self.df[self.df['PatientID'] == patient_id][feature].values[0] for feature in
                             self.config['columns']['clinical_features']]
        clinical_features = np.array(clinical_features).astype(np.float32)

        return image_stack, clinical_features, labels
