# -*- coding: utf-8 -*-

import yaml
import os


def save_config(config):
    """
    This function will make the config file that can be saved after excecuting the 
    main function.
    """
    
    current_dir = config['general']['resultsCurrentDirectory']
    out_name_config = config['Save']['filenames']['config_yaml']

    out_dir = os.path.join(current_dir, out_name_config)
    
    with open(out_dir,'w+') as yaml_file:
        yaml.dump(config,yaml_file,default_flow_style= False)
    
    return