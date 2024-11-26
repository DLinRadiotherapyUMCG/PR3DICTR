# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:25:03 2024

@author: HoekL02
"""

from src.visualization.Calibration import calibration_plot
from src.visualization.Confusion_matrix import confusion_matrix_bin
from src.visualization.ROC_curve import ROC_curve

def get_visualization(config, true, pred, label = ''):
    
    outpath = config['paths']['results']
    
    # if 'calibration' in config['visualization'].keys():
    calibration_plot(true,pred,config,label)
    
    # if 'confusion_matrix' in config['visualization'].keys():
    confusion_matrix_bin(true,pred,config,label)
        
    # if 'ROC_curve' in config['visualization'].keys():
    ROC_curve(true,pred,config,label)
    
    return