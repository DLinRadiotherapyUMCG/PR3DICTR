import csv
import os
from typing import Optional, List
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset.transforms.center_crop_3d import center_crop_3d
from src.constants import PATIENT_ID_COL_NAME

def get_delimiter(file_path: str) -> str:
    with open(file_path, 'r') as csvfile:
        delimiter = str(csv.Sniffer().sniff(csvfile.read()).delimiter)
        return delimiter



# Artificial end-point equality
def label_equalizer(df, config):
    
    ptnVar = PATIENT_ID_COL_NAME
    tox = config['columns']['labels']

    ratio = config['data']['equalizer']['ratio']
    method = config['data']['equalizer']['resamplingDirection']

    logging.info(f"Using label equaliser on training set. Direction = {method}, ratio = {ratio}")

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

    else:
        raise ValueError(f"resamplingDirection must be 'over' or 'under'. Currently: {method}")

