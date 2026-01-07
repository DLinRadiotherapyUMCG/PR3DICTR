# import global libraries
import captum
import torch.nn.functional as F
from tqdm import tqdm
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.attr import (
    IntegratedGradients,
    GuidedBackprop,
    GuidedGradCam,
    LayerGradCam,
    NoiseTunnel,
    GradientShap,)

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.visualization.plot_slices import plot_slices
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Local libraries enivironment loading
import os
import sys
from pathlib import Path
path_Project = Path(os.getcwd()).parent.parent
path_Git = path_Project.parent

def IncludeDirectory(path, index = 0, indexStop = -1):
    if(os.path.exists(path) and (index <= indexStop or indexStop == -1)):
        print(path)
        sys.path.append(path)
        children = os.listdir(path)
        for i in range(len(children)):
            pathChild = os.path.join(path,children[i])
            if(os.path.isdir(pathChild)):
                IncludeDirectory(pathChild, index + 1, indexStop)

# Activate the function
IncludeDirectory(path_Git,0,1)

# Import local libraries
from src.constants import DEVICE
from src.utils.set_random_seed import set_random_seed
from src.config_presets.tools.get_config import get_config
from src.config_presets.tools.load_config import load_config
from src.dataset.load_dataset import load_dataset, Complete_SanityCheck

from src.models.tools.load_model import load_model
from src.models.tools.get_classification_model import get_classification_model
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.utils.loss_func.get_loss_function import get_loss_function
from src.dataset.load_dataset import load_dataset
from src.dataset.get_transforms import get_transforms
from src.dataset.get_dataloader import make_dataloader

from src.utils.move_batch_to_device import move_batch_to_device
from src.constants import DEVICE

# ========================================================================================================================
trial_dir = r"/scratch/p315317/Project/DETOX_Lung_RP/Results/Habrok_ResNet18-02_2025/Trial_81"
config = get_config('DETOXLung_config')

test_dataset_source = config['data']['source']
print(f"Validating models on {test_dataset_source} test set")
# Get paths to folds and config from the experiment
folds = [f.path for f in os.scandir(trial_dir) if f.is_dir()] # This does asume all folders within a trial are folds
config_path = os.path.join(folds[0], config['saving']['filenames']['config_yaml']) 
print(config_path)

# First only do the first config
fold_config = load_config(config_path)

metricHandler = mainMetricHandler(fold_config)
loss_function = get_loss_function(fold_config)
train_transforms, val_transforms = get_transforms(fold_config)

# the dataset itself is loaded from the main config (this allows for using the external test set)
df_train_val, df_test = load_dataset(config)
test_loader, metadata = make_dataloader(config, df_test, val_transforms, validation_mode=True)

model = get_classification_model(fold_config, metadata, save_summary=False)
model.cuda()
model = load_model(fold_config, model)
#DEVICE = torch.device('cuda:1')
model = ModelInputWrapper(model)
#model.to(DEVICE) 

fold_dir = folds[0]
config['general']['resultsCurrentDirectory'] = fold_dir + r'/' # fold results saves to fold

# Now get the attention map (GradCam) options
attr = GuidedGradCam(model, model.module.encoder.backbone.block3[0])

# put the model into evaluation mode ! this saves some GPU space
model.eval()

def sigmoid(x):
    return 1/(1+np.exp(-x))

label = "RP_OneYear"
Sigmoid = np.vectorize(sigmoid)
index = 0
indexTarget = 5
indexSkipped = []

# Make the directory where the attention maps are created
config['general']['resultsCurrentDirectory'] = config['general']['resultsCurrentDirectory'] + "AttentionMaps" r'/'
os.makedirs(config['general']['resultsCurrentDirectory'], exist_ok=True)

with torch.no_grad():
    for i, batch in tqdm(test_loader):
        inputs, features, targets = move_batch_to_device(batch, DEVICE)
        # Load data, send it to the GPU
        #inputs, features, label_list = (
        #    data[0].cuda(),
        #    data[1].cuda(),
        #    data[2].cuda(),
        #    )

        # group the image inputs and the feature inputs into a tuple (this for the ModelInputWrapper)
        model_input = (torch.tensor(inputs), features) 

        # most methods require a baseline input. This is a fairly important choice, typically it is an array of zeros, but it can also be an array of pure noise or the mean of the dataset
        # these baselines are used to calculate the integral of the gradients, and they can have an effect on how the attention maps look (e.g. an all-zero baseline will lead to attention that highlights the white parts of the CT !)
        baseline_input = (torch.randn_like(torch.tensor(inputs)), torch.randn_like(features))  # random noise input
        #baseline_input = (torch.zeros_like(torch.tensor(inputs)), torch.zeros_like(features))  # all zeroes input
        

        # target index is the index of the class that we want to calculate the attention map for (i.e. the index of the value in the model's output tensor)
        # for single-toxicty models, I think it's just 0
        target_idx = 0
        # in Daniel's model code, this additional forward argument is used to make the model output in the shape of a tensor
        # normally the output is a dictionary {"Toxcity_A" : 1234, "Toxicity_B" : 5678, ...}, but for the attention maps, we need a tensor  [of shape (1, num_classes), I think]
        additional_forward_args = (1) 

        # the number of steps used by the approximation method (for integrated gradients, which uses 50 by default), it seems that more steps = more GPU useage
        n_steps = 1200
        batch_size = 16

        """
        For binary classification models:
        """
        # calculate the attention map: returns a tuple of two tensors, one tensor with the image attributions, and one with the clinical features attributions. They should have the same dimensions as the original input tensors.
        if(attr is IntegratedGradients):
            (image_attention, clinical_features_attention) = attr.attribute(model_input, baseline_input, additional_forward_args= additional_forward_args, n_steps=n_steps, internal_batch_size=batch_size)
        elif(attr is GuidedGradCam):
            (image_attention, clinical_features_attention) = attr.attribute(model_input, target=0, additional_forward_args= additional_forward_args, interpolate_mode="trilinear", attribute_to_layer_input=False)
        
        inputs_np = inputs.detach().cpu().numpy()
        image_attention_np = image_attention.detach().cpu().numpy()

        CT = inputs_np[0][0]
        RTDOSE = inputs_np[0][1]
        RTSTRUCT = inputs_np[0][2]

        kernel = np.array([0.25,0.5,1,0.5,0.25])

        plotting_rows_dicts = [
                    {
                        "Label": "CT_RTDOSE",
                        "CT": CT,
                        "RTDOSE": RTDOSE,
                    },
                    {
                        "Label": "Dose_attention",
                        "CT": CT,
                        #"RTDOSE": RTDOSE,
                        "Attention" : gaussian_filter(image_attention_np[0][1], sigma=2)
                    }
                ]

        slices= []
        dim = CT.shape[0]
        data = range(int(2*dim/6),int(4*dim/6),20)
        for sliceID in data:
            slices.append(sliceID)

        fig, axes = plot_slices(plotting_rows_dicts, slices, RT_region="LUNG", title=f"GuidedGradCam attention maps for patient with RP")

        for axes1 in axes:
            for axes2 in axes1:
                axes2.set_axis_off()
        path_AttentionMapPatient = config['general']['resultsCurrentDirectory'] + f"Ptn_{index}.png"
        fig.savefig(path_AttentionMapPatient)
                
        index +=1 
