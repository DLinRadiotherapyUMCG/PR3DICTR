# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def loss_plot(loss_t,loss_v, n_epochs, config):
    
    plt.figure()
    plt.plot(list(range(n_epochs)),loss_t)
    plt.plot(list(range(n_epochs)),loss_v)
    plt.savefig(config['paths']['results'] + 'loss.png')
    plt.clf()
    plt.close('all')
    
    return