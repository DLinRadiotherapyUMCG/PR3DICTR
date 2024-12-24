import csv
import os
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset.transforms.center_crop_3d import center_crop_3d
from src.constants import PATIENT_ID_COL_NAME

def get_delimiter(file_path: str) -> str:
    with open(file_path, 'r') as csvfile:
        delimiter = str(csv.Sniffer().sniff(csvfile.read()).delimiter)
        return delimiter


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
# def data_split(df, config, split = [.7,.15,.15], seed = 8):
#     """
#     Splits complete dataset [rows,columns] (rows = patients, columns = variables)
#     into three subsets: train, validation and test
    
#     """        
#     np.random.seed(seed)
#     tox = config['columns']['labels'] #'RP' 

#     # Train and Test split
#     stratifyItems = config['data']['stratifyList']
#     print("Stratify items: ",stratifyItems)
#     combinedColumn = []
#     for i in range(len(stratifyItems)):
#         if(len(combinedColumn) == 0):
#             combinedColumn = df[[stratifyItems[i]]].astype(str).to_numpy()
#         else:
#             combinedColumn = combinedColumn + df[[stratifyItems[i]]].astype(str).to_numpy()
#     stratList = combinedColumn
#     groupingList = numberGroup(stratList)
#     train_val, test = train_test_split(df, test_size = split[2], random_state=seed, stratify=groupingList)
    
#     # Train and validation split
#     split_trainval = split[1]/(split[1] + split[0])
#     combinedColumn = []
#     for i in range(len(stratifyItems)):
#         if(len(combinedColumn) == 0):
#             combinedColumn = train_val[[stratifyItems[i]]].astype(str).to_numpy()
#         else:
#             combinedColumn = combinedColumn + train_val[[stratifyItems[i]]].astype(str).to_numpy()
#     stratList = combinedColumn
#     groupingList = numberGroup(stratList)
#     train, validation = train_test_split(train_val, test_size = split_trainval, random_state=seed, stratify=groupingList)  
#     if(config['data']['equalizer']['isEnabled']):
#         train = label_equalizer(train, config)
#     return train, validation, test

# Artificial end-point equality
def label_equalizer(df, config):
    
    ptnVar = PATIENT_ID_COL_NAME
    tox = config['columns']['labels']

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
    
    if method == 'over':
        #startPercentage = sum(y == uniqueValues[1])/sum(y == uniqueValues[0]
        for i in range(len(df)):
            rowsIncluded.append(df.iloc[i].values)
        for i in range(len(instance_labels)):
            rangeIncluded = freq_max-instance_labels[i]
            if rangeIncluded > 0:
                for j in range(10000000):
                    percentageFound = (instance_labels[i] + j)/(len(y)+j)
                    if percentageFound >= ratio:
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
    
    elif method == 'under':
        for i in range(len(instance_labels)):
            r_indx = np.where(y_c == np.unique(y_c)[i])[0]
            np.random.shuffle(r_indx)            
            indexRange = freq_min
            totalCases = instance_labels[i]
            if totalCases > len(X)/2:
                index =  -1
                for j in range(totalCases):                   
                    ratioFound = (totalCases-j)/(len(X)-j)
                    if ratioFound <= (1-ratio):
                        index = j
                        break
                if index != -1:
                    indexRange = totalCases - index
            
            rowFound = df.iloc[r_indx[:indexRange]]

            for j in range(len(rowFound)):
                rowsIncluded.append(rowFound.iloc[j].values)
        # Create new dataframe
        df_new = pd.DataFrame(rowsIncluded, columns=columnHeaders)
        return df_new
    
    