import os
import toml


class ConfigObject:
    """
    A class object that can be used to store the configuration parameters of a model/experiment.
    Different copies of this can be passed around the functions of the main model training code to allow for
    different hyperparameters to be used when training different models simultaneously (e.g. for hyperparameter tuning).
    Args:
        cfg: a config object (the default config.py file)
    """

    def __init__(self, cfg):
        # get all of the variables in the config.py file and set them as 
        # attributes of this object (whenever we make a new instance of this class)
        for attr_name in dir(cfg):
            if not attr_name.startswith("__"):
                setattr(self, attr_name, getattr(cfg, attr_name))


def make_config_object(config_file, hp_param_config = None):
    # creates a cfg object, using the parameters defined in config.py
    #import config as config_file
    cfg = ConfigObject(config_file)

    # if we are using the best config, we need to overwrite the parameters that are different for this run
    if cfg.use_best_config:
        # load the best parameters for this model from the .toml file
        best_config = load_model_best_config(cfg, cfg.model_name)

        for attr_name, new_value in best_config.items():
            # cfg is an object, so we can set its attributes using setattr
            if new_value == "NONE": new_value = None                 # NOTE: workaround for lack of NoneType in toml

            setattr(cfg, attr_name, new_value)

    # if hyperparameter tuning, overwrite any parameters in the cfg that are different for this run of hp_param_config
    if hp_param_config is not None:
        # hp_param_config is a dictonary, so get the (key, value) pairs
        for attr_name, new_value in hp_param_config.items():
            # cfg is an object, so we can set its attributes using setattr
            setattr(cfg, attr_name, new_value)
            print(attr_name, new_value)
            #pass

    if cfg.use_umcg is False:
        #if not use_umcg:
        cfg.features_dl = [x.replace("W01", "BSL") for x in cfg.features_dl]

    return cfg



def load_model_best_config(config, tox: str):
    """
    Load the config file for the toxicity.
    :param tox:
    :return: Config
    """
    config_path = os.path.join('models','configs', tox + '.toml')
    # TODO: best configs folder dir in config file
    # e.g. config_path = os.path.join(config.best_configs_dir, tox + '.toml')

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise ValueError('Config file not found. Please check the toxicity_configs folder.')

    # Load the config file
    with open(config_path, 'r') as f:
        imported_config = toml.load(f)

    return imported_config