import csv
import os
from typing import Optional, List
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    X = df[ptnVar].to_numpy() # Patients 
    y = df[tox].to_numpy()  # lables
        
    columnHeaders = df.columns
    
    rowsIncluded = []
    y_c = y.copy()
    
    instance_labels =[]
    uniqueValues = np.unique(y)
    C_labels = len(uniqueValues)
        
    # Get labels were there is an postive label, so an occurence per class 
    for i in range(C_labels):
        instance_labels.append(np.sum(y == uniqueValues[i]))
        
    # calculate the min and max occurence instances
    freq_max = np.max(instance_labels)    
    freq_min = np.min(instance_labels)
    
    # Oversample by getting random patients untill the percentages equal out
    if method == 'over':
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
                    rowFound = df.iloc[r_indx].values
                    rowsIncluded.append(rowFound)
            # Create new dataframe with oversampled data  
        df_new = pd.DataFrame(rowsIncluded, columns=columnHeaders)        
        return df_new
    
    # Undersample by removing random patients untill the percentages equal out 
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

