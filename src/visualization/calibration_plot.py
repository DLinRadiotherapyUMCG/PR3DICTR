# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def calibration_plot(true,pred,config, label):
    
    
    n_bins = config['visualizations']['calibration']['num_bins']
    prob_true, prob_pred = calibration_curve(true,pred, n_bins=n_bins)
    
    plt.figure()
    plt.scatter(prob_pred,prob_true)
    plt.plot(prob_pred,prob_true,'-')
    plt.plot([0,1],[0,1],'--', c = 'k')
    plt.title('Calibration plot ' + label)
    plt.xlabel("Predicted probability (mean confidence)")
    plt.ylabel("True occurance rate")
    plt.savefig(config['paths']['results'] + 'calibration.png')
    plt.clf()
    plt.close('all')
    
    return
