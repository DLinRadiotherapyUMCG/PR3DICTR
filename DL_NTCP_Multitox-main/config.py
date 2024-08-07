import os
import sys
sys.path.append('./data_preproc')
import math
import pathlib
import torch
import pandas as pd
from datetime import datetime

import data_preproc.data_preproc_config as data_preproc_config
from data_preproc.data_preproc_functions import create_folder_if_not_exists, get_all_combinations, sort_human

# Whether to perform quick run for checking workability of code or not
perform_test_run = False
# Whether to perform hyperparameter tuning or not
hyperparameter_tuning_mode = False
# Whether to disable the training code (i.e. only test a pretrained model) 
test_model_mode = False
# Whether to use the test set at all (i.e. during model training or cross validation)
use_test_set = True

use_mean_loss_early_stopping = True
use_layer_freezing = False

train_CITOR_models = True

# Basic config
use_umcg = True
torch_version = torch.__version__
seed = 542
nr_of_decimals = 3


# Set directory contexts
if hyperparameter_tuning_mode and os.name == 'nt': 
    # the directories get messed up if using RayTune on windows
    # this is a workaround to make sure it reads the correct directories when looking for data
    #root_path = "//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Daniel MacRae/DL_NTCP_Multitox"
    #root_path = "C:/Users/danie/Desktop/FSE 23-34/Master's Project/DL_NTCP_Multitox"
    #root_path = "C:/Users/Daniel (7781D3A3)/Documents/DL_NTCP_Multitox"
    
    #root_path = r"c:\Users\S.P.M. de Vette\OneDrive - UMCG\Documents\DL_NTCP_Multitox"
    root_path = "C:/Users/d.c.macrae/Documents/DL_NTCP_Multitox"
    #root_path = pathlib.PureWindowsPath(root_path).as_posix()
else: 
    root_path = os.getcwd() 


models_dir = os.path.join(root_path, 'models')
optimizers_dir = os.path.join(root_path, 'optimizers')
data_preproc_dir = os.path.join(root_path, 'data_preproc')
save_root_dir = os.path.join(root_path, 'datasets')
data_folder = "MT_dataset" if use_umcg else "MDACC_dataset"
data_dir = os.path.join(save_root_dir, data_folder)
patients_data_dir = os.path.join(data_dir, 'patients')  # NEW MULTITOX # Can simply use the full dataset (the patients not in the csv are not loaded in the dataloader)
exp_root_dir = os.path.join(root_path, 'experiments', 'TRP_DenseNet121_m2 101') 
#exp_root_dir = os.path.join(root_path, 'experiments', 'calibration_plots')
if not hyperparameter_tuning_mode:
    create_folder_if_not_exists(exp_root_dir)
exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(exp_root_dir, exp_name)
exp_src_dir = os.path.join(exp_dir, 'src')
exp_models_dir = os.path.join(exp_src_dir, 'models')
exp_optimizers_dir = os.path.join(exp_src_dir, 'optimizers')
exp_data_preproc_dir = os.path.join(exp_src_dir, 'data_preproc')
exp_figures_dir = os.path.join(exp_dir, 'figures')
temp_model_weights_dir = os.path.join(exp_dir, 'tmp_model_weights')
filename_features_csv = "patients_all_features.csv"
filename_stratified_sampling_test_csv = 'stratified_sampling_test_542.csv'  # NEW MULTITOX  # test / train_val split csv
# filename_stratified_sampling_test_csv = 'per_toxicity_csv_542/Xerostomia_M06_542.csv'  # NOTE: SINGLE TOX MODEL  # test / train_val split csv
filename_stratified_sampling_full_csv = 'stratified_sampling_full_542.csv'  # NEW MULTITOX # test / val / train split csv
if not use_umcg:
    filename_stratified_sampling_test_csv = "MDACC_features_present.csv"
filename_train_val_test_patient_ids_json = 'train_val_test_patient_ids.json'
filename_results_csv = 'results.csv'
filename_results_overview_csv = 'results_ensemble.csv'
filename_i_outputs_csv = '{i}_outputs.csv'
filename_i_outputs_external_csv = '{i}_outputs_external.csv'
filename_lr_finder_png = 'lr_finder.png'
filename_model_txt = 'model.txt'
filename_best_model_pth = 'best_model.pth'
filename_best_optimizer_pth = 'best_optimizer.pth'
filename_best_scheduler_pth = 'best_scheduler.pth'
filename_results_png = 'results.png'

if use_umcg:
   patient_id_length = 7
   filename_exclude_patients_csv = 'exclude_patients_umcg.csv'
else:
   patient_id_length = 10
   filename_exclude_patients_csv = 'exclude_patients_mdacc.csv'


# Decide which device we want to run on
gpu_condition = torch.cuda.is_available()
device = torch.device('cuda:0') if gpu_condition else torch.device('cpu')
num_GPUs = torch.cuda.device_count() if gpu_condition else 0
cudnn_benchmark = False  # True if gpu_condition else False  # `True` will be faster, but potentially at cost of reproducibility
reproducible = False
# This flag allows us to enable the inbuilt CuDNN auto-tuner to find the best algorithm to use for our hardware. 
# Only enable if input sizes of our network do not vary.


# Stratified sampling config
# Patients to exclude (a manual list, for those patients not already in the exclude_patients_umcg.csv file):
exclude_list_manual = ["3573430", "5610140"]
# also include the csv of excluded patients  
exclude_list_csv_patient_IDs = [str(item).zfill(patient_id_length) for item in 
                                pd.read_csv(os.path.join(data_dir, filename_exclude_patients_csv), 
                                            delimiter=';', index_col=0).index]
# combine the two lists
exclude_list = exclude_list_manual + exclude_list_csv_patient_IDs
## (Clinical) columns to keep
main_columns = data_preproc_config.clinical_features #["PatientID", "Sex", "Age", "CT+C_available", "CT_Artefact", "Loctum2", "Photons", "Smoking", "T_stage", "N_stage", "P16", "WHO"]

# which toxicities we're modelling  
target_toxicities = ["Xerostomia", "Aspiration", "Sticky", "Taste", "Dysphagia"]   
#target_toxicities = [ "Xerostomia", "Dysphagia"]   

                        # ["Xerostomia", "Aspiration", "Sticky", "Taste", "Dysphagia"]
# timepoints we want to use
baseline_timepoints = ["W01"] if use_umcg else ["BSL"]  # which baseline to use ["BSL", "W01"]
endpoint_timepoints = ["M06"]    # what the endpoint time we're trying to predict is
                               # ["M06", "M12", "M18"]    
                                # can do: ["M06", "M12", "M18", "M24"]  but theres no dysphgia column for M24... so you'll get an error :/

## automatically generate the column names: (e.g. "Xerostomia_W01" or "Aspiration_M06")
toxicity_baseline_features = get_all_combinations(target_toxicities, baseline_timepoints, spacer="_")
toxicity_endpoint_features = get_all_combinations(target_toxicities, endpoint_timepoints, spacer="_")

# any features that we might need (i.e. a list that combines all of the features of the Xero, Taste, Dysphagia, and logistic regression NTCP models)
# any patient that is missing these gets dropped from the dataset
all_input_features = main_columns + toxicity_baseline_features #+ toxicity_endpoint_features

# Data config
test_frac = 0.2  # training-internal_validation-test split. The same test set will be used for Cross-Validation.
val_frac = 0.2  
train_frac = 1 - test_frac - val_frac  
sampling_type = 'stratified'  # ['random', 'stratified']. Method for dataset splitting.
perform_stratified_sampling_full = True  # (Stratified Sampling). Whether or not to recreate stratified_sampling_full.csv.
# Note: if stratified_sampling_full.csv does not exist, then we will perform stratified sampling to create the file.
strata_groups = ['CT+C', 'CT_Artefact', 'Photons', 'Loctum2'] # + toxicity_endpoint_features  # NEW MULTITOX: # (Stratified Sampling). Note: order does not matter.
split_col = 'Split'  # (Stratified Sampling). Column of the stratified sampling outcome ('train', 'val', 'test').
cv_folds = 5  # (Cross-Validation) If cv_folds=1, then perform train-val-test-split.
cv_type = "random" #'stratified'  # (Stratified CV, only if cv_folds > 1) None | 'stratified'. Stratification is performed on endpoint value.
dataset_type = 'cache'  # 'standard' | 'cache' | 'persistent'. If None, then 'standard'.
# Cache: caches data in RAM storage. Persistent: caches data in disk storage instead of RAM storage.
cache_rate = 1.0  # (dataset_type='cache')
runtime_cache = 'process' # (dataset_type='cache') None | 'process' | 'thread'. (recommend 'process' to prevent memory leak)
num_workers = 8 # 4 * num_GPUs  # `4 * num_GPUs` (dataset_type='cache' or 'persistent')
persistent_workers = True if num_workers > 0 else False 
to_device = False  # if num_workers > 0 else True  # Whether or not to apply `ToDeviced()` in Dataset' transforms.
pin_memory = True if num_workers > 0 else False  # Whether or not to pin memory (to GPU) in DataLoader.
# See load_data.py. Suggestion: set True on HPC, set False on local computer.
cache_dir = os.path.join(root_path, 'persistent_cache')  # (only if dataset_type='persistent')
dataloader_type = 'standard'  # 'standard' | 'thread'. If None, then 'standard'. Thread: leverages the separate thread
# to execute preprocessing to avoid unnecessary IPC between multiple workers of DataLoader.
use_sampler = False  # Whether to use WeightedRandomSampler or not.
dataloader_drop_last = False  # (standard, thread)
segmentation_structures = data_preproc_config.structures_uncompound_list
valid_endpoint_values = [0, 1]  # NEW MULTITOX. Values other than these values will be skipped during training,
# e.g., set missing values to -1.

# Data preprocessing config: load_data.py 
image_keys = ['ct', 'rtdose', 'segmentation_map']  # Do not change
use_multihot_segmap = False
concat_key = 'ct_dose_seg'  # Do not change
rand_cropping_size = [96, 96, 96]  # (REDUNDANT)  # OLD [100, 100, 100]  # (only for training data)
input_size = [96, 96, 96]  # OLD [100, 100, 100]  # if rand_cropping_size==input_size, then no resizing will be applied
resize_mode = 'area'  # (only if perform_resize=True). Algorithm used for upsampling.
# CT
ct_a_min = -200  # OLD data_preproc_config.ct_min if data_preproc_config.perform_clipping else None
ct_a_max = 400  # OLD data_preproc_config.ct_max if data_preproc_config.perform_clipping else None
ct_b_min = 0.0
ct_b_max = 1.0
ct_clip = True
# RTDOSE
rtdose_a_min = 0
rtdose_a_max = 8000
rtdose_b_min = 0.0
rtdose_b_max = 1.0
rtdose_clip = True
# Segmentation_map
# Advice: use the order as ['background'] + data_preproc_config.structures_uncompound_list = \
# ['background', 'parotis_li', 'parotis_re', 'submandibularis_li', 'submandibularis_re',
#  'crico', 'thyroid', 'mandible', 'glotticarea', 'oralcavity_ext', 'supraglottic',
#  'buccalmucosa_li', 'buccalmucosa_re', 'pcm_inf', 'pcm_med', 'pcm_sup', 'esophagus_cerv']
#  Note: `+ 1` in `seg_orig_labels` because of `background`
seg_orig_labels = [x for x in range(len(data_preproc_config.structures_uncompound_list) + 1)]  # No need to change
#seg_target_labels = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0.1, 0, 0.1, 0.1, 0, 0, 0, 0]  # [0] + [1] * len(data_preproc_config.structures_uncompound_list)
seg_target_labels = [0] + [1] * len(data_preproc_config.structures_uncompound_list)   #   DANIEL
segmentation_map_clip = True
# Data augmentation config (only if perform_data_aug=True)
perform_data_aug = True
data_aug_p = 0.5  # Probability of a data augmentation transform: data_aug_p=0 is no data aug.
data_aug_strength = 2  # Strength of data augmentation: strength=0 is no data aug (except flipping).
# AugMix
perform_augmix = False
mixture_width = 3  # 3 (default)
mixture_depth = [1, 3]  # [1, 3] (default)
augmix_strength = 3
# MixUp
perform_mixup = True
mixup_alpha = 3

# Model config
model_name = "TransRP_DenseNet121_m2" # "SimpleTransRP_ResNet18_m2" # "TransRP_ResNet18_m2" #"convnext_large" # 'dcnn_pooling'  # 'resnet_lrelu', 'dcnn_lrelu', 'resnet_dcnn_lrelu', 'dcnn_pooling', "ViT", "Swin3D"
                        # 'convnext', "TransRP"
use_best_config = False
endpoint_list = sort_human(toxicity_endpoint_features)
predict_CT_contrast = False
siamese_encoder = False
skip_connection = False
if predict_CT_contrast: endpoint_list.append("CT+C")
# endpoint_list = ['Parotid_L_meandose', 'Parotid_R_meandose', 'Submandibular_L_meandose', 'Submandibular_R_meandose',
#                 'PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'Crico_meandose', 'Supraglottic_meandose', 'OralCavity_Ext_meandose', 'BuccalMucosa_L_meandose',
#                 'BuccalMucosa_R_meandose', 'TongueTop_meandose', 'External_meandose', 'PCM_meandose', 'Esophagus_Cerv_meandose']
features_dl = ['Sex', 'Age', # "Chemotherapy",
               'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
               'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
               'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
                'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
               'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg',
                'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
                #  'T0','T1','T2','T3','T4','Tx',
                #  'N0','N1','N2','N3',
                #  'P16_Negatief','P16_Niet_bepaald','P16_Positief',
                #  'Smoking_1',
                ]  # ['Photons']  # NEW MULTITOX: these columns are used by all endpoints, so make sure that there are no missing values!
#features_dl = []
# features_dl = ['Sex', 'Age', # "Chemotherapy",
#                'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg',
#                 'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
#                 'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
#                 'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
#                 'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
#                 'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
#                 'T0','T1','T2','T3','T4','Tx',
#                 'N0','N1','N2','N3',
#                 'P16_Negatief','P16_Niet_bepaald','P16_Positief',
#                 'Smoking_1',
#                 ]
# features_dl =  ['Sex', 'Age', 
#                'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg',
#                 'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
#                 'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
#                 'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
#                 'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
#                 'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
#                 'T0','T1','T2','T3','T4','Tx',
#                 'N0','N1','N2','N3',
#                 'P16_Negatief','P16_Niet_bepaald','P16_Positief',
#                 'Smoking_1',
#                 ]
if not use_umcg:
    features_dl = [x.replace("W01", "BSL") for x in features_dl]
# features_dl = [] 
# [] | data_preproc_config.features  # Should contain columns in features.csv.
resnet_shortcut = 'B'  # (resnet_original) 'A', 'B'. Pretrained resnet10_original has 'B', resnet18_original has 'A'.
n_input_channels = 3  # CT, RTDOSE and Segmentation_map
conv_block_layers = 1 # [1,2] (dcnn_pooling) number of layers per block
filters = [64, 128, 256, 512]
kernel_sizes = [3, 3, 3, 3, 3]
strides = [[2]*3, [2]*3, [2]*3, [2]*3]  # strides>2 currently not supported.
#strides = [[1]*3, [1]*3, [1]*3, [1]*3, [1]*3]  
normalization = 'batch'  # ['batch', 'instance', 'none']
resnet_model_depth = 18  # 10 (default), 18, 34, 50, 101, 152.  # NOTE: this overrides the previous 3 variables
densenet_model_depth = 121  # 121 (default), 169, 201, 264.  # NOTE: this overrides the previous 3 variables
conv1_t_size = 5  # 7 (default) # ResNet
conv1_t_stride = 2
# Usually len(kernel_sizes) == len(filters), but len(kernel_sizes) > len(filters) is often allowed, but then
# kernel_sizes[:len(filters)] will be used. Similarly for strides.
pad_value = 0  # (Padding) Value used for padding.
lrelu_alpha = 0.01  # (LeakyReLU) Negative slope.
pooling_conv_filters = None  # Either int or None (i.e. no pooling conv before flattening).
perform_pooling = False  # Whether to perform (Avg)Pooling or not. If pooling_conv_filters is not None, then
pooling = 'avg'  # ['max', 'avg', 'none']. If 'none', then pooling will not be applied.
pooling_size = 2 # integer 2 = [2,2,2] pooling
# (Avg)Pooling will not be applied.
convnext_patch_size = 4
convnext_kernel_size = 7
convnext_drop_first_block = False
vit_dim = 128   # (ViT) last dimension of output tensor after linear transformation
vit_depth = 4  # (ViT) number of transformer blocks
vit_heads = 8 # (ViT) number of heads in the multi-attention layer
vit_mlp_dim = 128 # (ViT) dimension of the feedforward layer
vit_patch_size = 8  # (ViT) size of the patches [4,8,16]
vit_dim_head = 128 # (ViT) dimension of the heads (dim ** -0.5)
swin_patch_size = 4  # (Swin Transformer) size of the patches
swin_window_size = 8 # (Swin Transformer) size of the window
swin_depths = [2,2,18,2]  # (Swin Transformer) number of layers in each stage
swin_num_heads = [3,6,12,24] # (Swin Transformer) number of heads in each stage
swin_embed_dim = int(96 + 96/2)  # (Swin) size of embedding dimension (num feature maps)
# LINEAR LAYERS:
clinical_variables_position = 1  # (Only if len(features_dl) > 0.)  1 | 2 | 3 | ... The position is where the `clinical_linear_units`
# will be connected into the shared linear layers, e.g., 1 means that the clinical variables MLP will be added to the first shared linear layer
# can do up to len(linear_units) + 1, which means the clinical features get added to the first non-shared layer
clinical_variables_linear_units = []  # (Only if len(features_dl) > 0.) None | list of ints
linear_units = [256]  # NEW MULTITOX. SHARED linear layers. None | list of ints
dropout_p = 0.1 # Should have the same length as `linear_units`. None | list of ints
linear_units_endpoint = [64]  # NEW MULTITOX. NON-SHARED linear layers (endpoint specific)
use_bias = True
num_ohe_classes = 1 # Size of One-Hot Encoding output (per toxicity endpoint).  IMPORTANT: define `label_weights` such that len(label_weights) == num_ohe_classes.
ECE_n_bins = 10

# Optimization config
pretrained_path = None # "C:/Users/d.c.macrae/Documents/DL_NTCP_Multitox/experiments/Autoencoder_0/resnet_lrelu" # "HP_experiments/TL_Baseline_Weights"
# None # "C:/Users/S.P.M. de Vette/OneDrive - UMCG/Documents/DL_NTCP_Multitox/experiments/Dose_Predictor_Weights" # None # "experiments/weights"  # None, path_to_and_including_pth (e.g. 'MedicalNet/pretrain/resnet_10_23dataset.pth')
transfer_learning_mode = False # True  # True  # Transfer learning: when True: replaces the output_head with a new one
weight_init_name = 'kaiming_uniform'  # [None, 'kaiming_uniform', 'uniform', 'xavier_uniform', 'kaiming_normal',
# 'normal', 'xavier_normal', 'orthogonal']. If None, then PyTorch's default (i.e. 'kaiming_uniform', but with
# a = math.sqrt(5)). Kaiming works well if the network has (Leaky) P(ReLU) activations.
kaiming_a = math.sqrt(5)  # [math.sqrt(5) (default), lrelu_alpha]. Only used when kaiming_nonlinearity = 'leaky_relu'.
kaiming_mode = 'fan_in'  # ['fan_in' (default), 'fan_out'].
kaiming_nonlinearity = 'leaky_relu'  # ['leaky_relu' (default), 'relu', 'selu', 'linear']. When using 
# weight_init_name = kaiming_normal for initialisation with SELU activations, then nonlinearity='linear' should be used 
# instead of nonlinearity='selu' in order to get Self-Normalizing Neural Networks.
gain = 0  # [0 (default), torch.nn.init.calculate_gain('leaky_relu', lrelu_alpha)].
optimizer_name = 'sgd'  # ['acc_sgd', 'ada_belief', 'ada_bound', 'ada_bound_w', 'ada_hessian', 'ada_mod', 'adam',
# 'adam_w', 'apollo', 'diff_grad', 'madgrad', 'novo_grad', 'qh_adam', 'qhm', 'r_adam', 'ranger_21', 'ranger_qh',
# 'rmsprop', 'pid', 'sgd', 'sgd_w', 'swats', 'yogi']
momentum = 0.9  # 0 (default), 0.85, 0.9 (common), 0.95. For optimizer_name in ['rmsprop', 'sgd', 'sgd_w'].
weight_decay = 0.01  # 0 (default), 0.01 (for optimizer_name in ['adam_w', 'sgd_w']). L2 regularization penalty.
hessian_power = 1.0  # (AdaHessian)
use_lookahead = False  # (Lookahead)
lookahead_k = 5  # (Lookahead) 5 (default), 10.
lookahead_alpha = 0.5  # (Lookahead) 0.5 (default), 0.8.

loss_function_name = 'bce'    # 'bce' | 'focal' | 'asymmetric' | 'hill' |'SPLC' | "MLLSC" | 'mse'
# [None, 'bce' (num_classes = 1), 'cross_entropy' (num_classes = 2), 'cross_entropy',
# 'dice', 'f1', 'ranking', 'soft_auc', 'custom']. Note: if 'bce', then also change label_weights to list of 1 element.
# Note: model output should be logits, i.e. NO sigmoid() (BCE) nor softmax() (CE) applied.
# if these next two are in use, the loss function is wrapped by WeightedLossWrapper to apply (one or both of) these weights
# if None, then no weighting is applied, else, set the weights to a list of length endpoint_list
# # TODO : they also both get normalised to sum to 1 !
label_weights = False #  [2,3,4,1,2]  #  the weighting of the losses per endpoint node (i.e. per toxicity)
label_pos_weights = False #  whether to apply automatic weighting of the positive instances in each class
alpha = 0.25  # (FocalLoss) 0.25 (default)
gamma = 2.0     # (FocalLoss, HillLoss (MLLSC)) 2 (default)
gamma_neg = 0 # (asymmetric loss)
gamma_pos = 2 # (asymmetric loss)
margin = 0.2 # (FocalMarginLoss (=0 is just Focal loss), HillLoss, )
lamb = 1.5 # (HillLoss) lambda value
tau = 0.4 # (SPLC loss)
mllsc_variant = "focal"
tau_pos = 0.6 # (MLLSC loss)
tau_neg = 0.55 # (MLLSC loss)
change_epoch = 2 # (SPLC, MLLSC loss)
auxiliary_loss_func_name = None # "ESD"  #  "SB-ECE" # None  # "SB-ECE" | "ESD" | None
SB_ECE_n_bins = 15
SB_ECE_temperature = 0.005
loss_reduction = 'none' # Daniel: do not change this
label_smoothing = 0  # (LabelSmoothing) If 0, then no label smoothing will be applied.
scheduler_name = None # 'cosine'  # [None, 'cosine', 'cyclic', 'exponential']. If None, then no scheduler will be applied.
grad_max_norm = 1 # 1.0 # None # (GradientClipping) Maximum norm of gradients. If None, then no grad clipping will be applied.
lr = 0.001  # 0.0004750810162102798 

# manual schedulers (e.g. 'manual_lr' (DEPRECATED), see misc.get_scheduler()).
T_0restart = 40  # (CosineAnnealingWarmRestarts) Number of epochs until a restart.
T_mult = 1  # (CosineAnnealingWarmRestarts) A factor increases after a restart. Default: 1.
eta_min = 1e-10  # (CosineAnnealingWarmRestarts) Minimum learning rate. Default: 0.
base_lr = 1e-6  # (CylicLR, only if perform_lr_finder = False, see get_scheduler()) Minimum and starting learning rate.
max_lr = 0.00001 # 1e-4  # (CylicLR, only if perform_lr_finder = False, see get_scheduler()) Maximum learning rate.
step_size_up = 79 * 15  # (CyclicLR) Number of training iterations in the increasing half of a cycle.
sched_gamma = 0.95  # (ExponentialLR, StepLR) Multiplicative factor of learning rate decay every epoch (ExponentialLR) or
# step_size (StepLR).
step_size = 15  # (StepLR) Decays the learning rate by gamma every step_size epochs.
factor_lr = 0.5  # (Reduce_LR) Factor by which the learning rate will be updated: new_lr = lr * factor.
min_lr = 1e-8  # (Reduce_LR) Minimum learning rate allowed.

# Training config
max_epochs = 100
batch_size = 12
eval_interval = 1
patience = 10  # (EarlyStopping): stop training after this number of consecutive epochs without
# internal validation improvements.
loss_tolerance = 0 
sub_ensemble = False
sub_ensemble_N = 10

# Plotting config
plot_interval = 50
max_nr_images_per_interval = 1
figsize = (12, 18)


# To test if the size of the training set influences model performance
subsample_train_set = False
subsample_train_set_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Logistic regression models config
submodels_features_list_dict = {
    "Aspiration" : [
        ['PCM_Med_meandose', 'Aspiration_W01_Een_beetje', 'Aspiration_W01_Nogal_Heel_erg'] # Aspiration (no submodels)
    ],
    "Dysphagia" : [
        ['OralCavity_Ext_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 'Loctum2_Pharynx', 'Loctum2_Larynx'],  # Dysphagia submodel 1
        ['PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 'Loctum2_Pharynx', 'Loctum2_Larynx']          # Dysphagia submodel 2
    ], 
    "Sticky" : [
        ['Submandibular_meandose', 'Parotid_meandose', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg'],  # Sticky (no submodels)
    ], 
    "Taste" : [
        ['Parotid_meandose', 'OralCavity_Ext_meandose', 'Age'], # Taste (no submodels)
    ], 
    "Xerostomia" : [
        ['Parotid_meandose', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'],    # Xerostomia submodel 1
        ['Submandibular_meandose', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'],    # Xerostomia submodel 2
    ]
}

features_lr_list_dict = {
    "Aspiration" : ['PCM_Med_meandose', 'Aspiration_W01_Een_beetje', 'Aspiration_W01_Nogal_Heel_erg'], 
    "Dysphagia"  : ['OralCavity_Ext_meandose', 'PCM_Sup_meandose', 'PCM_Med_meandose', 'PCM_Inf_meandose', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 'Loctum2_Pharynx', 'Loctum2_Larynx'], 
    "Sticky"     : ['Submandibular_meandose', 'Parotid_meandose', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg'],  
    "Taste"      : ['Parotid_meandose', 'OralCavity_Ext_meandose', 'Age'], 
    "Xerostomia" : ['Submandibular_meandose', 'Parotid_meandose', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'] 
}

lr_coefficients_list_dict = {
    "Aspiration" : None,
    "Dysphagia" : None,
    "Sticky" : None,
    "Taste" : None,
    "Xerostomia" : None,
}







if perform_test_run:
    n_samples = 500
    max_epochs = 1
    patience = 5
    test_frac = 0.1
    val_frac = 0.3
    train_frac = 1 - val_frac - test_frac
    cv_folds = 2
    batch_size = 8
    lr = 1e-3
    
    plot_interval = 1
    max_nr_images_per_interval = 5

    # dataloader stuff
    num_workers = 8
    persistent_workers = True
    perform_data_aug = False
    perform_augmix = False
    #runtime_cache = None




# stuff I will never change
# Interpolation modes: 'trilinear'/'bilinear' | 'nearest'.
ct_interpol_mode_3d = 'trilinear'
rtdose_interpol_mode_3d = 'trilinear'
segmentation_interpol_mode_3d = 'nearest'  
ct_dose_seg_interpol_mode_3d = 'trilinear'
ct_interpol_mode_2d = 'bilinear'
rtdose_interpol_mode_2d = 'bilinear'
segmentation_interpol_mode_2d = 'nearest'
ct_dose_seg_interpol_mode_2d = 'bilinear'


if model_name == "MLP":
    perform_augmix = False
    perform_mixup = False
    perform_data_aug = False
    batch_size = 128