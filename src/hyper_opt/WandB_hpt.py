import wandb
from src.utils.fileHandler import create_file, create_folder, create_textfile

def WandB_is_enabled(config: dict):
    return config['hyperparam_tuning']['WandB']['IsEnabled']

def WandB_create_study(config: dict, dict_params, groupName = None):
    create_folder(config['general']['resultsCurrentDirectory'])
    if (WandB_is_enabled(config) == True):
        wandb.init(
            project=config['general']['experiment_name'],
            dir=config['general']['resultsCurrentDirectory'][:-1],
            config = dict_params,
            group = groupName,
            reinit = True
        ) 

def WandB_log(config: dict, results, epoch):   # UpdateStudy
    if (WandB_is_enabled(config) == True):
        infoSend = dict()
        keys = list(results.keys())
        for i in range(len(keys)):
            infoSend[keys[i]] = results[keys[i]]
        print(infoSend)
        wandb.log(data = infoSend, step = epoch)
        #listMetrics = config['hyperparam_tuning']['WandB']['MetricsMonitor']
        #if(len(results) != len(listMetrics)):
        #    raise Exception(f"The WandB metrics list ({len(listMetrics)}) and results ({len(results)}) do not match in length!")
        #for i in range(len(results)):



def login(config: dict) -> None:   
    if WandB_is_enabled(config) == True:
        # Make sure that you are logged in your python environment  
        # Login with account
        wandb.login(key=config["hyperparam_tuning"]["WandB"]["API_Key"])            
    return



def WandB_stop(config: dict) -> None:
    if (WandB_is_enabled(config) == True):
        wandb.finish(quiet=True)




def initialise_WandB_group(config: dict, project_name: str, groupName = None) -> None:
    """
    A WandB group is a collection of runs that are grouped together. 
    This is useful for grouping together K-folds of the same trial or model.
    """
    #wandb.login(key=config["hyperparam_tuning"]["WandB"]["API_Key"])  

    if (WandB_is_enabled(config) == True):
    #if config['hyperparam_tuning']['WandB']['IsEnabled']:
        wandb.init(
            project = project_name,
            group = groupName,
            dir=config['general']['resultsCurrentDirectory'][:-1],
            config = config,
            reinit = True
        ) 

    #print(project_name)
    #print(groupName)
    


def update_WandB_summary_table(config, best_log_dict):
    """

    """
    if (WandB_is_enabled(config) == True):
        # supress the WandB summary-maker 
        # (it somehow insists on using the most recent log, not the 'best', even if you define what that is)
        for key in best_log_dict.keys():
            wandb.define_metric(key, summary="none")

        # update the best log. This appears in the 'summary' tab in WandB (the table of results)
        wandb.run.summary.update(best_log_dict)
