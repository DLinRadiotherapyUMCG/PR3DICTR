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

def get_delimiter(file_path: str) -> str:
    with open(file_path, 'r') as csvfile:
        delimiter = str(csv.Sniffer().sniff(csvfile.read()).delimiter)
        return delimiter

def get_umcg_n(path, ptnsRejected = None):   
    umcg_n = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    
    if(ptnsRejected != None):
        umcgNmbrs = []
        for i in range(len(umcg_n)):
            if((int(umcg_n[i]) in ptnsRejected) == False):
                umcgNmbrs.append(umcg_n[i])
        umcg_n = umcgNmbrs
    return np.array(umcg_n)


def unique(list1):
    unique_list = []
    # traverse for all elements
    for i in range(len(list1)):
        if(list1[i] not in unique_list):
            unique_list.append(list1[i])
    return unique_list

def numberGroup(list1):
    uniqueValues = unique(list1)
    groupList = []
    
    for i in range(len(list1)):
        for j in range(len(uniqueValues)):
            if(list1[i] == uniqueValues[j]):
                groupList.append([j])
                break
    return groupList


# Splitter
def Data_split(df, config, split = [.7,.15,.15], seed = 8):
    """
    Splits complete dataset [rows,columns] (rows = patients, columns = variables)
    into three subsets: train, validation and test
    
    """        
    np.random.seed(seed)
    tox = 'RP' 

    # Train and Test split
    stratifyItems = config['data']['stratifyList']
    combinedColumn = []
    for i in range(len(stratifyItems)):
        if(len(combinedColumn) == 0):
            combinedColumn = df[[stratifyItems[i]]].astype(str).to_numpy()
        else:
            combinedColumn = combinedColumn + df[[stratifyItems[i]]].astype(str).to_numpy()
    stratList = combinedColumn
    groupingList = numberGroup(stratList)
    train_val, test = train_test_split(df, test_size = split[2], random_state=seed, stratify=groupingList)
    
    # Train and validation split
    split_trainval = split[1]/(split[1] + split[0])
    combinedColumn = []
    for i in range(len(stratifyItems)):
        if(len(combinedColumn) == 0):
            combinedColumn = train_val[[stratifyItems[i]]].astype(str).to_numpy()
        else:
            combinedColumn = combinedColumn + train_val[[stratifyItems[i]]].astype(str).to_numpy()
    stratList = combinedColumn
    groupingList = numberGroup(stratList)
    train, validation = train_test_split(train_val, test_size = split_trainval, random_state=seed, stratify=groupingList)  
    if(config['data']['equalizer']['isEnabled']):
        previousSize = train.shape[0]
        train = label_equalizer(train, config)
    return train, validation, test

# Artificial end-point equality
def label_equalizer(df, config):
    
    ptnVar = config['data']['patientVar']
    tox = 'RP' 

    ratio = config['data']['equalizer']['ratio']
    method = config['data']['equalizer']['resamplingDirection']

    X = df[ptnVar].to_numpy()
    y = df[tox].to_numpy()
        
    columnHeaders = df.columns
    
    rowsIncluded = []
    X_c = X.copy()
    y_c = y.copy()
    
    instance_labels =[]
    uniqueValues = np.unique(y)
    C_labels = len(uniqueValues)
        
    for i in range(C_labels):
        instance_labels.append(np.sum(y == uniqueValues[i]))
        
    freq_max = np.max(instance_labels)    
    freq_min = np.min(instance_labels)
    
    if method == 'up':
        #startPercentage = sum(y == uniqueValues[1])/sum(y == uniqueValues[0]
        for i in range(len(df)):
            rowsIncluded.append(df.iloc[i].values)
        for i in range(len(instance_labels)):
            rangeIncluded = freq_max-instance_labels[i]
            if(rangeIncluded > 0):
                for j in range(10000000):
                    percentageFound = (instance_labels[i] + j)/(len(y)+j)
                    if(percentageFound >= ratio):
                        rangeIncluded = j
                        break
                for ii in range(rangeIncluded):
                    r_indx = np.where(y_c == np.unique(y_c)[i])[0][np.random.randint(0,instance_labels[i]-1)]
                    #X = np.append(X,X_c[r_indx])
                    #y = np.append(y,y_c[r_indx])
                    rowFound = df.iloc[r_indx].values
                    rowsIncluded.append(rowFound)
            # Create new dataframe
        
        df_new = pd.DataFrame(rowsIncluded, columns=columnHeaders)        
        return df_new
    
    
    if method == 'down':
        for i in range(len(instance_labels)):
            r_indx = np.where(y_c == np.unique(y_c)[i])[0]
            np.random.shuffle(r_indx)            
            indexRange = freq_min
            totalCases = instance_labels[i]
            if(totalCases > len(X)/2):
                index =  -1
                for j in range(totalCases):                   
                    ratioFound = (totalCases-j)/(len(X)-j)
                    if(ratioFound <= 1-ratio):
                        index = j
                        break
                if(index != -1):
                    indexRange = totalCases - index
            
            rowFound = df.iloc[r_indx[:indexRange]]
            for j in range(len(rowFound)):
                rowsIncluded.append(rowFound.iloc[j].values)
        # Create new dataframe
        df_new = pd.DataFrame(rowsIncluded, columns=columnHeaders)
        return df_new
    
    if method == "None":
        return df
    


def preprocess_image_stack(image_stack, config):
    """
    Preprocesses a single image.
    :param image_stack:
    :param config:
    :return:
    """
    if(config['preprocessing']['isEnabled']):
        # Standardization
        ct_scaler = ScaleIntensityRange(a_min=config['preprocessing']['ct']['a_min'],
                                        a_max=config['preprocessing']['ct']['a_max'],
                                        b_min=config['preprocessing']['ct']['b_min'],
                                        b_max=config['preprocessing']['ct']['b_max'],
                                        clip=config['preprocessing']['ct']['clip'],
                                        dtype=np.float32)
        rtdose_scaler = ScaleIntensityRange(a_min=config['preprocessing']['rtdose']['a_min'],
                                            a_max=config['preprocessing']['rtdose']['a_max'],
                                            b_min=config['preprocessing']['rtdose']['b_min'],
                                            b_max=config['preprocessing']['rtdose']['b_max'],
                                            clip=config['preprocessing']['rtdose']['clip'],
                                            dtype=np.float32)
        segmentation_map_transform = MapLabelValue(
            orig_labels=[index for index, x in enumerate(config['preprocessing']['segmentation_map']['target_labels'])],
            target_labels=config['preprocessing']['segmentation_map']['target_labels'])

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
            transforms.append(RandSpatialCrop(roi_size=config['preprocessing']['crop_shape'], random_size=False, random_center=True))
        
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
            transform.set_random_state(seed=config['seed'])

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
        if self.config['preprocessing']['crop']:
            cropped_image_stack = np.empty((image_stack.shape[0], *self.config['preprocessing']['crop_shape']))
            for i in range(image_stack.shape[0]):
                cropped_image_stack[i] = center_crop_3d(image_stack[i], self.config['preprocessing']['crop_shape'])
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
            transform.set_random_state(seed=config['seed'])

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
        if self.config['data']['augmentation']['isEnabled']:
            image_stack = self.transforms(image_stack)

        # If cropping is enabled, crop the image
        if self.config['preprocessing']['crop']:
            print("Cropping")
            cropped_image_stack = np.empty((image_stack.shape[0], *self.config['preprocessing']['crop_shape']))
            for i in range(image_stack.shape[0]):
                cropped_image_stack[i] = center_crop_3d(image_stack[i], self.config['preprocessing']['crop_shape'])
            image_stack = cropped_image_stack.astype(np.float32)

        # Get the label
        label = np.array([self.df[self.df['PatientID'] == patient_id][self.label_column].values[0]]).astype(np.float32)

        # Get the clinical features
        clinical_features = [self.df[self.df['PatientID'] == patient_id][feature].values[0] for feature in
                             self.config['columns']['clinical_features']]
        clinical_features = np.array(clinical_features).astype(np.float32)

        return image_stack, clinical_features, label
