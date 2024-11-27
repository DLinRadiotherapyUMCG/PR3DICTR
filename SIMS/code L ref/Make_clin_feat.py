# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 07:54:25 2024

@author: Luuk.vanderHoek
"""

import numpy as np
import pandas as pd

from data_preparation import get_endpoint_data

features_data = r'C:\Users\Luuk\Desktop\project1\data\patients_all_features.xlsx'

df = pd.read_excel(features_data)

X = df['PatientID'].to_numpy()

base_tox = get_endpoint_data(X, time_point_idx = 0, imputed = True, binary = False, tox = '')
W1_tox = get_endpoint_data(X, time_point_idx = 1, imputed = True, binary = False, tox = '')

xer_w1_niet = []
xer_w1_beetje = []
xer_w1_nogal_heelerg = []
sex = []
age = []

out = r'C:\Users\Luuk\Desktop\project1\data\clin_features.xlsx'

for P_id in X:
    if P_id in base_tox[0]:
        tox = base_tox[1][np.where(base_tox[0] == P_id)]        
    if P_id in W1_tox[0]:
        tox = base_tox[1][np.where(base_tox[0] == P_id)]

    if tox == 0:
        xer_w1_niet.append(1)
        xer_w1_beetje.append(0)
        xer_w1_nogal_heelerg.append(0)
    elif tox == 1:
        xer_w1_niet.append(0)
        xer_w1_beetje.append(1)
        xer_w1_nogal_heelerg.append(0)
    elif tox == 2 or tox == 3:
        xer_w1_niet.append(0)
        xer_w1_beetje.append(0)
        xer_w1_nogal_heelerg.append(1)
    else:
        xer_w1_niet.append(0)
        xer_w1_beetje.append(0)
        xer_w1_nogal_heelerg.append(0)       
        
    sex_temp = df['Sex'][np.where(df['PatientID'] == P_id)[0][0]]
    if sex_temp == 'Man':
        sex.append(1)
    elif sex_temp == 'Vrouw':
        sex.append(0)
    else:
        sex.append(1)
        
    age.append(df['Age'][np.where(df['PatientID'] == P_id)[0][0]]/100)
    
    
clin_fea_df = pd.DataFrame({'PatientID':X,'Xer_w1_0':xer_w1_niet, 'Xer_w1_1': xer_w1_beetje, 'Xer_W1_2_3':xer_w1_nogal_heelerg,
                            'sex':sex,'age':age})
clin_fea_df.to_excel(out)