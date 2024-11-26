# -*- coding: utf-8 -*-

import yaml

def make_config(config):
    """
    This function will make the config that can be saved after excecuting the 
    main function.
    """
    
    out_name_config = config['yaml']['out_path']
    
    with open(out_name_config,'w+') as yaml_file:
        yaml.dump(config,yaml_file,default_flow_style= False)
    
    return