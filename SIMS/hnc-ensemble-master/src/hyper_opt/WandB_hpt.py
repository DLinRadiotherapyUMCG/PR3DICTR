import wandb

def WandEnabled(config):
    return config['hyperparam_tuning']['WandB']['IsEnabled']

def CreateStudy(config, dict_params, groupName = None):
    if(WandEnabled(config) == True):
        wandb.init(
            project=config['hyperparam_tuning']['ProjectName'],
            config = dict_params,
            group = groupName,
            reinit = True
        ) 

def UpdateStudy(config, results, epoch):
    if(WandEnabled(config) == True):
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


def WandB_initalise(config):   
    if(WandEnabled(config) == True):
        # Make sure that you are logged in your python environment (check WandB setup tutorial)    
        # Login with account
        wandb.login()            
    return

def WandB_stop(config):
    if(WandEnabled(config) == True):
        wandb.finish()