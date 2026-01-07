import wandb
import logging
import copy

def is_WandB_enabled(config: dict):
    return config['hyperparam_tuning']['WandB']['isEnabled']

def WandB_log(config: dict, results : dict, epoch: int = None):   # UpdateStudy
    """
    Logs an epoch to WandB
    """
    if is_WandB_enabled(config):
        wandb.log(data = copy.deepcopy(results), step = epoch)

def login(config: dict) -> None:  
    """
    Logs into WandB using the API key in the config
    """ 
    if is_WandB_enabled(config):
        logging.info("Logging into Weights and Biases...")
        # Login with account
        wandb.login(key=config["hyperparam_tuning"]["WandB"]["API_Key"])   
        wandb.Settings(quiet=True)  # Suppress WandB output  
    else:
        logging.info("Weights and Biases is not enabled in the config, skipping login.")  

def stop_WandB_trial(config: dict) -> None:
    """
    Finishes a WandB run (e.g. a trial).
    """
    if is_WandB_enabled(config):
        wandb.finish()

def initialise_WandB_group(config: dict, project_name: str, groupName = None, config_for_wandb = None) -> None:
    """
    A WandB group is a collection of runs that are grouped together. 
    This is useful for grouping together K-folds of the same trial or model.
    """

    dir = config['general']['resultsCurrentDirectory']
    if dir.endswith('/'):
        dir = dir[:-1]

    if is_WandB_enabled(config):
        wandb.init(
            project = project_name,
            group = groupName,
            dir = dir,
            config = config_for_wandb,
            reinit = True
        ) 
    
def update_WandB_summary_table(config : dict, best_log_dict : dict):
    """
    A helper function to enforce that the WandB summary table contains information about the 'best' epoch,
    and not of the 'last' epoch.
    """

    if is_WandB_enabled(config):
        # supress the WandB summary-maker 
        # (it somehow insists on using the most recent log, not the 'best', even if you define what that is)
        for key in best_log_dict.keys():
            wandb.define_metric(key, summary="none")

        # update the best log. This appears in the 'summary' tab in WandB (the table of results)
        wandb.run.summary.update(best_log_dict)


