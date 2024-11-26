import os
import torch
import numpy as np
import multiprocessing
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from typing import Dict, Optional, Any
import config as cfg

def define_number_of_GPUS_per_process(num_GPUs):
    if num_GPUs == 0: # no GPU
        return 0
    else:
        if max_concurrent_processes > 1:
            if num_GPUs > 1: # more than 1 GPU: 1 GPU per process
                return min(1, num_GPUs/max_concurrent_processes)
            if num_GPUs == 1: # 1 GPU: share it evenly between each process
                return 1/max_concurrent_processes # subtract a small number to avoid rounding errors
        else: # 1 process, can just have the whole GPU
            return num_GPUs

seed = cfg.seed

# set paths (automatically processed from the main config file)
storage_path, exp_name = os.path.split(cfg.exp_root_dir)

tune_experiment_dir = cfg.exp_root_dir #os.path.join(storage_path, exp_name)

# set the number of GPUs and CPUs per process
num_GPUs = torch.cuda.device_count()
num_CPU_cores = multiprocessing.cpu_count()
max_concurrent_processes = 4
num_GPUs_per_process = 0.5 # define_number_of_GPUS_per_process(num_GPUs)
num_CPUs_per_process = 16 # num_CPU_cores/max_concurrent_processes

# max_concurrent_processes = 2
# num_GPUs_per_process = 1
# num_CPUs_per_process = 6 # num_CPU_cores/max_concurrent_processes

# allowing experiments to pick up from the last checkpoint
resume_errored = False,    # 0.  # if the experiment crashed, should we resume it?
restart_errored = False,   # 1.   # if crashed, should we restart it from scratch?
resume_unfinished = True  # NOTE: Keep this one True to allow the whole experiment to resume from the last checkpoint !!


# which metrics to optimise:
# NOTE: could also minimise the loss?
metrics = "mean_val_AUC"
modes = 'max'
# multi-objective optimisation:
metrics = [f"mean_{endpoint}_val_AUC" for endpoint in cfg.endpoint_list] + ['mean_val_AUC'] + [f"mean_{endpoint}_val_ECE" for endpoint in cfg.endpoint_list]
modes = ['max' if "AUC" in x else 'min' for x in metrics]


num_HP_trials = 1000  # number of hyperparameter trials to run
num_random_trials_start = 80

# possible values for the hyperparameters (that ray-tune/optuna can sample from)
batch_size_vals = [8, 16, 32, 64]
relu_alpha_vals = [0.01, 0.1, 0.2, 0.3]
num_filter_vals = [4, 8, 16, 32, 64]
kernel_size_vals = [2, 3, 4, 5, 6, 7]
stride_vals = [2]  # strides >2 not supported by the current model [TODO: implement it, see how to calculate the size of the flattening layer]
#stride_vals = [1]

model_names = ['resnet_lrelu', 'dcnn_lrelu', 'resnet_dcnn_lrelu', 'resnext_lrelu']
resnet_model_depths = [10, 18, 34, 50, 101, 152]
resnet_model_depths = [10]


linear_unit_vals = [8, 16, 32, 64, 128, 256, 512]
dropout_p_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
stride_dim = cfg.n_input_channels

learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

# the parameter space of hyperparameters that we want to search
def define_by_run_func_OLD(trial) -> Optional[Dict[str, Any]]:
    """
    Define-by-run function to create the search space. 
    Returns a dict of parameters to be used for the current trial, and is built to handle conditional hyperparameters.
        e.g. if the model has 2 conv layers, then the kernel_sizes and strides will be a list of 2 elements, etc.
    source: https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/optuna_define_by_run_example.py
    """
    # pick the model name
    #model_name = trial.suggest_categorical("model_name", model_names)
    model_name = cfg.model_name # 'ViT' # 'dcnn_pooling' # 'resnet_lrelu' #'dcnn_pooling' 


    if model_name == "efficientnetv2":
        #model_name = trial.suggest_categorical("model_name", ["efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_xl"])
        model_name = trial.suggest_categorical("model_name", ["efficientnetv2_xs", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_xl"])
    
    # general hyperparameters    
    trial.suggest_categorical("batch_size", batch_size_vals)
    trial.suggest_categorical("lrelu_alpha", relu_alpha_vals)

    # number of convolutional layers in the model backbone
    if model_name == 'resnet_lrelu':
        num_conv_layers = trial.suggest_int("num_conv_layers", 3, 4)
        trial.suggest_categorical("resnet_model_depth", resnet_model_depths)
        #trial.suggest_categorical("resnet_shortcut", ["A", "B"])
        trial.suggest_categorical("conv1_t_size", [3, 5, 7])
        trial.suggest_categorical("conv1_t_stride", [1, 2])
    elif model_name == 'resnext_lrelu':
        trial.suggest_categorical("resnet_model_depth", [10, 18, 34, 50, 101])
        num_conv_layers = 1
        stride_vals = [1, 2]
        #trial.suggest_categorical("normalization", ['batch'])
    elif model_name == "ViT":
        num_conv_layers = 0
        trial.suggest_categorical("vit_dim", [16, 32, 64, 128, 256, 512])
        trial.suggest_categorical("vit_depth", [4, 6, 8, 10])
        trial.suggest_categorical("vit_heads", [4, 8, 16, 32])
        trial.suggest_categorical("vit_mlp_dim", [16, 32, 64, 128, 256, 512])
        trial.suggest_categorical("vit_patch_size", [8, 16])
        trial.suggest_categorical("vit_dim_head", [16, 32, 64, 128, 256, 512])
    elif "efficientnet" in model_name:
        pass
    elif "convnext" in model_name.lower():
        trial.suggest_categorical("model_name", ['ConvNeXt_tiny','ConvNeXt_small','ConvNeXt_base', 'ConvNeXt_large'])
    elif "swin3d" in model_name.lower():
        trial.suggest_categorical("model_name", ['Swin3D_tiny','Swin3D_small','Swin3D_base', 'Swin3D_large'])
        #trial.suggest_categorical("model_name", ['ConvNeXt_base','ConvNeXt_large'])
    elif "medvit" in model_name.lower():
        trial.suggest_categorical("model_name", ['MedViT_small','MedViT_base','MedViT_large'])
    else:
        num_conv_layers = trial.suggest_int("num_conv_layers", 5, 6)
        trial.suggest_categorical("normalization", ['batch'])
        trial.suggest_categorical("conv_block_layers", [1, 2])
        trial.suggest_categorical("pooling_size", [2])
        trial.suggest_categorical("pooling", ['max', 'avg'])
    if model_name == 'dcnn_pooling':
        stride_vals = [1]
    
    elif "convnext" in model_name.lower():
        convnext_kernel_size = trial.suggest_categorical("convnext_kernel_size", [3,5,7])
        convnext_patch_size = trial.suggest_categorical("convnext_patch_size", [4,8,16])
        convnext_drop_first_block = trial.suggest_categorical("convnext_drop_first_block", [True, False])
    elif "medvit" in model_name.lower() or "swin3d" in model_name.lower():
        pass
    elif model_name != "Vit" and "efficientnet" not in model_name and 'resnet_lrelu' not in model_name:
        # sample 'num_conv_layers' times from the list of possible filter, kernel_size, and stride values
        if model_name != 'resnext_lrelu':
            filters = [trial.suggest_categorical(f"filters_{i+1}", num_filter_vals) for i in range(num_conv_layers)]
        kernel_sizes = [trial.suggest_categorical(f"kernel_sizes_{i+1}", kernel_size_vals) for i in range(num_conv_layers)]
        strides = [[trial.suggest_categorical(f"strides_{i+1}", stride_vals)]*stride_dim for i in range(num_conv_layers)]
    
    # number of shared fully connected layers
    num_shared_fc_layers = trial.suggest_int("num_shared_fc_layers", 1, 4)
    linear_units = [trial.suggest_categorical(f"shared_linear_layer_{i+1}", linear_unit_vals) for i in range(num_shared_fc_layers)]
    trial.suggest_categorical("dropout_p", dropout_p_vals)

    # number of fc layers for the clinical variables
    clinical_variables_position = trial.suggest_int("clinical_variables_position", 1, 2)
    num_clinical_fc_layers = trial.suggest_int("num_clinical_fc_layers", 0, 4)
    clinical_variables_linear_units = [trial.suggest_categorical(f"clinical_variables_linear_layer_{i+1}", linear_unit_vals) for i in range(num_clinical_fc_layers)]

    # number of non-shared fully connected layers
    num_nonshared_endpoint_layers = trial.suggest_int("num_endpoint_fc_layers", 0, 4)
    endpoint_linear_units = [trial.suggest_categorical(f"endpoint_layer_{i+1}", linear_unit_vals) for i in range(num_nonshared_endpoint_layers)]

    # Data augmentation
    perform_data_aug = trial.suggest_categorical("perform_data_aug", [True])
    if perform_data_aug:
        trial.suggest_float("data_aug_p", 0.1, 1, step=0.1)
        trial.suggest_float("data_aug_strength", 0, 5, step=1)
    if cfg.perform_mixup:                                        # NOTE: MixUp
        trial.suggest_float("mixup_alpha", 0, 5, step=0.2)
    # perform_augmix = trial.suggest_categorical("perform_augmix", [False])
    # if perform_augmix:
    #     trial.suggest_int("mixture_width", 1, 5)
    #     min_mixture_depth = trial.suggest_int("min_mixture_depth", 1,4)
    #     max_mixture_depth = trial.suggest_int("max_mixture_depth", min_mixture_depth, 4)
    #     mixture_depth = [min_mixture_depth, max_mixture_depth] 
    #     trial.suggest_int("augmix_strength", 1,5)
    
    # training hyperparameters
    loss_function_name = trial.suggest_categorical("loss_function_name", ['bce', 'focal', 'asymmetric', 'hill']) # 'SPLC', 'MLLSC'
    if loss_function_name == 'focal':
        trial.suggest_float("alpha", 0, 1, step=0.1)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("margin", 0, 2, step=0.5)
    elif loss_function_name == 'asymmetric':
        trial.suggest_int("gamma_neg", 0, 10)
        trial.suggest_int("gamma_pos", 0, 6)
    #weigh_class_losses = trial.suggest_categorical("Weighted_Class_Loss", [True, False])
    elif loss_function_name == 'bce':
        trial.suggest_categorical("label_pos_weights", [True, False])
    elif loss_function_name == 'hill':
        trial.suggest_float("lamb", 0.5, 3, step=0.5)
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
    elif loss_function_name == 'SPLC':
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("tau", 0.3, 0.7, step=0.1)
        trial.suggest_int("change_epoch", 0, 5)  
    elif loss_function_name == 'MLLSC':
        trial.suggest_categorical("mllsc_variant", ["bce", "focal"])
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("tau_pos", 0.3, 0.7, step=0.05)
        trial.suggest_float("tau_neg", 0.3, 0.7, step=0.05)
        trial.suggest_int("change_epoch", 0, 5)  

    # Auxillary loss
    auxiliary_loss_func_name = trial.suggest_categorical("auxiliary_loss_func_name", ["SB-ECE", "ESD", None])
    if auxiliary_loss_func_name == "SB-ECE":
        trial.suggest_categorical("SB_ECE_n_bins", [10, 15, 20, 25])
        trial.suggest_categorical("SB_ECE_temperature", [0.01, 0.005, 0.001, 0.0005, 0.0001])
    # optimisers, schedulers, and learning rate
    optimizer_name = trial.suggest_categorical("optimizer_name", ['ada_belief', 'ada_bound', 'adam', 'rmsprop', 'sgd', 'yogi'])
    if optimizer_name in ['sgd', 'rmsprop']:
        trial.suggest_float("momentum", 0, 1)
    scheduler_name = trial.suggest_categorical("scheduler_name", [None, 'cosine', 'cyclic'])
    trial.suggest_categorical("lr", learning_rates)
    if scheduler_name == 'cosine':
        trial.suggest_int("T_0restart", 5, 100, step=5)
    elif scheduler_name == 'cyclic':
        trial.suggest_categorical("base_lr", [1e-6, 5e-6, 1e-5])
        trial.suggest_categorical("max_lr", [5e-5, 1e-4, 5e-4, 1e-3])
        trial.suggest_int("step_size_up", 100, 10000, step=100) 
    #trial.suggest_categorical("warmup_batches", [0, 1, 3, 5])
    

    
    # reformat those suggestions into a dict format thats suits the existing code
    param_space = {    
                    
                    'num_shared_fc_layers' : num_shared_fc_layers,
                    'linear_units' : linear_units, 
                    
                    'clinical_variables_position' : clinical_variables_position, 
                    'num_clinical_fc_layers' : num_clinical_fc_layers, 
                    'clinical_variables_linear_units' : clinical_variables_linear_units, 
                    'linear_units_endpoint' : endpoint_linear_units,

                }
    if model_name != "Vit" and "efficientnet" not in model_name and "convnext" not in model_name.lower() and (not "medvit" in model_name.lower()) \
    and (not "swin3d" in model_name.lower() and 'resnet_lrelu' not in model_name):
        if model_name != 'resnext_lrelu':
            param_space['filters'] = filters
        param_space['kernel_sizes'] = kernel_sizes
        param_space['strides'] = strides
    # add the conditional hyperparameters in the useable format
    #if perform_augmix: 
    #    param_space['mixture_depth'] = mixture_depth
    if loss_function_name != 'bce':
        param_space["label_pos_weights"] = False 
    
    return param_space


# time limit (in case a trial gets stuck or hangs somewhere, probably better to kill it)
max_hours = 5                       # how many hours
time_limit_s = max_hours * 60 * 60  # convert to seconds



if cfg.perform_test_run:
    max_concurrent_processes = 2
    num_GPUs_per_process = 0.5 #define_number_of_GPUS_per_process(num_GPUs)
    num_CPUs_per_process = 4 # 6 if num_GPUs > 0 else 8/max_concurrent_processes

    num_HP_trials = 50
    




features_dl1 = ['Sex', 'Age', # "Chemotherapy",
               'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg',
                'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
                'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
                'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
                'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
                'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
                ]
features_dl2 =  ['Sex', 'Age', "Chemotherapy",
               'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg',
                'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
                'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
                'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
                'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
                'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
                ]
features_dl3 =  ['Sex', 'Age', 
               'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg',
                'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
                'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
                'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
                'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
                'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
                'T0','T1','T2','T3','T4','Tx',
                'N0','N1','N2','N3',
                ]
features_dl4 =  ['Sex', 'Age', 
               'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg',
                'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
                'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
                'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
                'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
                'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
                'T0','T1','T2','T3','T4','Tx',
                'N0','N1','N2','N3',
                'P16_Negatief','P16_Niet_bepaald','P16_Positief',
                'Smoking_1',
                ]

DL_feature_combos = [features_dl1, features_dl3, features_dl4]



# the parameter space of hyperparameters that we want to search
def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
    """
    Define-by-run function to create the search space. 
    Returns a dict of parameters to be used for the current trial, and is built to handle conditional hyperparameters.
        e.g. if the model has 2 conv layers, then the kernel_sizes and strides will be a list of 2 elements, etc.
    source: https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/optuna_define_by_run_example.py
    """
    # pick the model name
    #model_name = trial.suggest_categorical("model_name", model_names)
    model_name = "TransRP_ResNet18_m2" # 'ViT' # 'dcnn_pooling' # 'resnet_lrelu' #'dcnn_pooling' 

    # general hyperparameters    
    trial.suggest_categorical("batch_size", [4, 8,16,32])
    trial.suggest_categorical("lrelu_alpha", relu_alpha_vals)
    #trial.suggest_int("features_dl", 0, 1)
    features_dl = DL_feature_combos[trial.suggest_int("features_dl", 0, 1)]

    # number of convolutional layers in the model backbone
    
    if "resnet" in model_name.lower():
        trial.suggest_categorical("resnet_model_depth", [10, 18, 34, 50, 101, 152])
        trial.suggest_categorical("conv1_t_size", [3, 5, 7])
        trial.suggest_categorical("conv1_t_stride", [2])
    else:
        trial.suggest_categorical("densenet_model_depth", [121, 169, 201, 264])
    

    # number of shared fully connected layers
    num_shared_fc_layers = trial.suggest_int("num_shared_fc_layers", 0, 3)
    linear_units = [trial.suggest_categorical(f"shared_linear_layer_{i+1}", linear_unit_vals) for i in range(num_shared_fc_layers)]
    trial.suggest_categorical("dropout_p", dropout_p_vals)

    # number of fc layers for the clinical variables
    # clinical_variables_position = trial.suggest_int("clinical_variables_position", 1, 2)
    # num_clinical_fc_layers = trial.suggest_int("num_clinical_fc_layers", 0, 4)
    # clinical_variables_linear_units = [trial.suggest_categorical(f"clinical_variables_linear_layer_{i+1}", linear_unit_vals) for i in range(num_clinical_fc_layers)]

    # number of non-shared fully connected layers
    num_nonshared_endpoint_layers = trial.suggest_int("num_endpoint_fc_layers", 0, 3)
    endpoint_linear_units = [trial.suggest_categorical(f"endpoint_layer_{i+1}", linear_unit_vals) for i in range(num_nonshared_endpoint_layers)]

    # Data augmentation
    perform_data_aug = trial.suggest_categorical("perform_data_aug", [True])
    if perform_data_aug:
        trial.suggest_float("data_aug_p", 0.1, 1, step=0.1)
        trial.suggest_float("data_aug_strength", 0, 5, step=1)
    if cfg.perform_mixup:                                        # NOTE: MixUp
        trial.suggest_float("mixup_alpha", 0, 5, step=0.2)
    
    # training hyperparameters
    loss_function_name = trial.suggest_categorical("loss_function_name", ['bce', 'focal', 'asymmetric', 'hill']) # 'SPLC', 'MLLSC'
    if loss_function_name == 'focal':
        trial.suggest_float("alpha", 0, 1, step=0.1)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("margin", 0, 2, step=0.5)
    elif loss_function_name == 'asymmetric':
        trial.suggest_int("gamma_neg", 0, 10)
        trial.suggest_int("gamma_pos", 0, 6)
    #weigh_class_losses = trial.suggest_categorical("Weighted_Class_Loss", [True, False])
    elif loss_function_name == 'bce':
        trial.suggest_categorical("label_pos_weights", [True, False])
    elif loss_function_name == 'hill':
        trial.suggest_float("lamb", 0.5, 3, step=0.5)
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
    elif loss_function_name == 'SPLC':
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("tau", 0.3, 0.7, step=0.1)
        trial.suggest_int("change_epoch", 0, 5)  
    elif loss_function_name == 'MLLSC':
        trial.suggest_categorical("mllsc_variant", ["bce", "focal"])
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("tau_pos", 0.3, 0.7, step=0.05)
        trial.suggest_float("tau_neg", 0.3, 0.7, step=0.05)
        trial.suggest_int("change_epoch", 0, 5)  

    # Auxillary loss
    auxiliary_loss_func_name = trial.suggest_categorical("auxiliary_loss_func_name", ["SB-ECE", "ESD", None])
    if auxiliary_loss_func_name == "SB-ECE":
        trial.suggest_categorical("SB_ECE_n_bins", [10, 15, 20, 25])
        trial.suggest_categorical("SB_ECE_temperature", [0.01, 0.005, 0.001, 0.0005, 0.0001])
    # optimisers, schedulers, and learning rate
    optimizer_name = trial.suggest_categorical("optimizer_name", ['ada_belief', 'ada_bound', 'adam', 'rmsprop', 'sgd', 'yogi'])
    if optimizer_name in ['sgd', 'rmsprop']:
        trial.suggest_float("momentum", 0, 1)
    scheduler_name = trial.suggest_categorical("scheduler_name", [None, 'cosine', 'cyclic'])
    trial.suggest_categorical("lr", learning_rates)
    if scheduler_name == 'cosine':
        trial.suggest_int("T_0restart", 5, 100, step=5)
    elif scheduler_name == 'cyclic':
        trial.suggest_categorical("base_lr", [1e-6, 5e-6, 1e-5])
        trial.suggest_categorical("max_lr", [5e-5, 1e-4, 5e-4, 1e-3])
        trial.suggest_int("step_size_up", 100, 10000, step=100) 
    #trial.suggest_categorical("warmup_batches", [0, 1, 3, 5])
    

    
    # reformat those suggestions into a dict format thats suits the existing code
    param_space = {    
                    'features_dl' : features_dl,
                    'num_shared_fc_layers' : num_shared_fc_layers,
                    'linear_units' : linear_units, 
                    
                    # 'clinical_variables_position' : clinical_variables_position, 
                    # 'num_clinical_fc_layers' : num_clinical_fc_layers, 
                    # 'clinical_variables_linear_units' : clinical_variables_linear_units, 
                    'linear_units_endpoint' : endpoint_linear_units,

                }
    
    # add the conditional hyperparameters in the useable format
    #if perform_augmix: 
    #    param_space['mixture_depth'] = mixture_depth
    if loss_function_name != 'bce':
        param_space["label_pos_weights"] = False 
    
    return param_space












# the parameter space of hyperparameters that we want to search
def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
    """
    Define-by-run function to create the search space. 
    Returns a dict of parameters to be used for the current trial, and is built to handle conditional hyperparameters.
        e.g. if the model has 2 conv layers, then the kernel_sizes and strides will be a list of 2 elements, etc.
    source: https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/optuna_define_by_run_example.py
    """
    # pick the model name
    #model_name = trial.suggest_categorical("model_name", model_names)
    model_name = "MLP" # 'ViT' # 'dcnn_pooling' # 'resnet_lrelu' #'dcnn_pooling' 
    linear_unit_vals = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # general hyperparameters    
    trial.suggest_categorical("batch_size", [4, 8,16,32, 64, 128])
    trial.suggest_categorical("lrelu_alpha", relu_alpha_vals)
    #trial.suggest_int("features_dl", 0, 1)
    features_dl = DL_feature_combos[trial.suggest_int("features_dl", 0, 1)]

    # number of shared fully connected layers
    num_shared_fc_layers = trial.suggest_int("num_shared_fc_layers", 0, 5)
    linear_units = [trial.suggest_categorical(f"shared_linear_layer_{i+1}", linear_unit_vals) for i in range(num_shared_fc_layers)]
    trial.suggest_categorical("dropout_p", dropout_p_vals)

    # number of non-shared fully connected layers
    num_nonshared_endpoint_layers = trial.suggest_int("num_endpoint_fc_layers", 0, 5)
    endpoint_linear_units = [trial.suggest_categorical(f"endpoint_layer_{i+1}", linear_unit_vals) for i in range(num_nonshared_endpoint_layers)]

    # Data augmentation
    trial.suggest_categorical("perform_data_aug", [False])
    trial.suggest_categorical("perform_mixup", [False])
    trial.suggest_categorical("perform_augmix", [False])
    
    # training hyperparameters
    loss_function_name = trial.suggest_categorical("loss_function_name", ['bce', 'focal', 'asymmetric', 'hill']) # 'SPLC', 'MLLSC'
    if loss_function_name == 'focal':
        trial.suggest_float("alpha", 0, 1, step=0.1)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("margin", 0, 2, step=0.5)
    elif loss_function_name == 'asymmetric':
        trial.suggest_int("gamma_neg", 0, 10)
        trial.suggest_int("gamma_pos", 0, 6)
    #weigh_class_losses = trial.suggest_categorical("Weighted_Class_Loss", [True, False])
    elif loss_function_name == 'bce':
        trial.suggest_categorical("label_pos_weights", [True, False])
    elif loss_function_name == 'hill':
        trial.suggest_float("lamb", 0.5, 3, step=0.5)
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
    elif loss_function_name == 'SPLC':
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("tau", 0.3, 0.7, step=0.1)
        trial.suggest_int("change_epoch", 0, 5)  
    elif loss_function_name == 'MLLSC':
        trial.suggest_categorical("mllsc_variant", ["bce", "focal"])
        trial.suggest_float("margin", 0.0, 3, step=0.2)
        trial.suggest_int("gamma", 0, 6)
        trial.suggest_float("tau_pos", 0.3, 0.7, step=0.05)
        trial.suggest_float("tau_neg", 0.3, 0.7, step=0.05)
        trial.suggest_int("change_epoch", 0, 5)  

    # Auxillary loss
    auxiliary_loss_func_name = trial.suggest_categorical("auxiliary_loss_func_name", ["SB-ECE", "ESD", None])
    if auxiliary_loss_func_name == "SB-ECE":
        trial.suggest_categorical("SB_ECE_n_bins", [10, 15, 20, 25])
        trial.suggest_categorical("SB_ECE_temperature", [0.01, 0.005, 0.001, 0.0005, 0.0001])
    # optimisers, schedulers, and learning rate
    optimizer_name = trial.suggest_categorical("optimizer_name", ['ada_belief', 'ada_bound', 'adam', 'rmsprop', 'sgd', 'yogi'])
    if optimizer_name in ['sgd', 'rmsprop']:
        trial.suggest_float("momentum", 0, 1)
    scheduler_name = trial.suggest_categorical("scheduler_name", [None, 'cosine', 'cyclic'])
    trial.suggest_categorical("lr", learning_rates)
    if scheduler_name == 'cosine':
        trial.suggest_int("T_0restart", 5, 100, step=5)
    elif scheduler_name == 'cyclic':
        trial.suggest_categorical("base_lr", [1e-6, 5e-6, 1e-5])
        trial.suggest_categorical("max_lr", [5e-5, 1e-4, 5e-4, 1e-3])
        trial.suggest_int("step_size_up", 100, 10000, step=100) 
    #trial.suggest_categorical("warmup_batches", [0, 1, 3, 5])
    

    # reformat those suggestions into a dict format thats suits the existing code
    param_space = {    
                    'features_dl' : features_dl,
                    'num_shared_fc_layers' : num_shared_fc_layers,
                    'linear_units' : linear_units, 
                    'linear_units_endpoint' : endpoint_linear_units,
                }
    
    # add the conditional hyperparameters in the useable format
    if loss_function_name != 'bce':
        param_space["label_pos_weights"] = False 
    
    return param_space

