import wandb
from src.utils.fileHandler import create_file, create_folder, create_textfile

def WandB_is_enabled(config):
    return config['hyperparam_tuning']['WandB']['IsEnabled']

def WandB_create_study(config, dict_params, groupName = None):
    create_folder(config['general']['resultsCurrentDirectory'])
    if (WandB_is_enabled(config) == True):
        wandb.init(
            project=config['general']['experiment_name'],
            dir=config['general']['resultsCurrentDirectory'][:-1],
            config = dict_params,
            group = groupName,
            reinit = True
        ) 

def WandB_log(config, results, epoch):   # UpdateStudy
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



def login(config):   
    if WandB_is_enabled(config) == True:
        # Make sure that you are logged in your python environment  
        # Login with account
        wandb.login(key=config["hyperparam_tuning"]["WandB"]["API_Key"])            
    return



def WandB_stop(config):
    if(WandB_is_enabled(config) == True):
        wandb.finish(quiet=True)




def single_WandB_group(config, project_name, groupName = None):
    #wandb.login(key=config["hyperparam_tuning"]["WandB"]["API_Key"])  

    #if(WandEnabled(config) == True):
    wandb.init(
        project = project_name,
        group = groupName,
        dir=config['general']['resultsCurrentDirectory'][:-1],
        config = config,
        reinit = True
    ) 

    print(project_name)
    print(groupName)
    