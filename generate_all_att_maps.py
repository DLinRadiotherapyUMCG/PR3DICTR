
# import global libraries
import captum
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.attr import (
    IntegratedGradients,
    GuidedBackprop,
    GuidedGradCam,
    LayerGradCam,
    NoiseTunnel,
    GradientShap,
    Saliency,
    LRP)

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
    GradientShap,
    Deconvolution,
    Saliency,
    LayerLRP,)
from captum.attr import LayerAttribution

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt


# Import local libraries
from src.constants import DEVICE
from src.utils.set_random_seed import set_random_seed
from src.models.tools.get_classification_model import get_classification_model
from src.config_presets.tools.get_config import get_config
from src.config_presets.tools.load_config import load_config
from src.dataset.load_dataset import load_dataset
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.get_transforms import get_transforms


DEVICE = torch.device('cuda:0')


attr_methods_dict = {
    #"IntegratedGradients": IntegratedGradients,
    #"GradCam++": LayerGradCam,
    #"GuidedGradCam": GuidedGradCam,
    #"GradCam": LayerGradCam,
    "Deconvolution" : Deconvolution,
}


def make_integrated_gradients_image(model, target_idx, inputs, additional_forward_args, baseline_input):
    attr = IntegratedGradients(model)
    #baseline_input_randn = torch.zeros_like(torch.tensor(inputs))
    image_attention = attr.attribute(inputs, target=target_idx, baselines=baseline_input, additional_forward_args=additional_forward_args, n_steps=100)

    return image_attention[0].squeeze().cpu().detach().numpy()


def make_gradcam_image(model, target_idx, inputs, additional_forward_args, model_layer, plusplus=True):
    gradcam = LayerGradCam(model, model_layer)
    image_attention = gradcam.attribute(inputs, target=target_idx, additional_forward_args=additional_forward_args, plusplus=plusplus, attribute_to_layer_input=False)

    # print("GC++", image_attention[0].shape)

    image_attention = LayerAttribution.interpolate(image_attention, inputs.shape[2:], interpolate_mode='trilinear')

    return image_attention[0].squeeze().cpu().detach().numpy()


def make_guided_gradcam_image(model, target_idx, inputs, additional_forward_args, model_layer):
    guidedgradcam = GuidedGradCam(model, model_layer)
    image_attention = guidedgradcam.attribute(inputs, target=target_idx, additional_forward_args=additional_forward_args, attribute_to_layer_input=False)

    return image_attention[0].squeeze().cpu().detach().numpy()

def make_deconvolution_image(model, target_idx, inputs, additional_forward_args):
    deconv = Deconvolution(model)
    image_attention = deconv.attribute(inputs, target=target_idx, additional_forward_args=additional_forward_args)

    return image_attention[0].squeeze().cpu().detach().numpy()


def main(experiment_dir, test_loader, metadata, config, output_dir):

    K_fold_folders = [f.name for f in os.scandir(experiment_dir) if f.is_dir()]
    print("Folders in experiment_dir:", K_fold_folders)



    for fold_idx, fold_folder in enumerate(K_fold_folders):
        fold_dir = os.path.join(experiment_dir, fold_folder)
        
        pathModelWeigths = os.path.join(fold_dir, filename_model_weights)
        model = get_classification_model(config, metadata, False)
        model.load_state_dict(torch.load(pathModelWeigths))
        print("Successfully loaded model and datasets")

        # wrap the model with the ModelInputWrapper
        model = ModelInputWrapper(model)

        model.to(DEVICE)     # send the model to the GPU

        # set the model to evaluation mode
        model.eval()


        
        modules_dict_list = {
            "conv1": model.module.encoder.conv1,
            "block0": model.module.encoder.backbone.block0[0],
            "block1": model.module.encoder.backbone.block1[0],
            "block2": model.module.encoder.backbone.block2[0],
            "block3": model.module.encoder.backbone.block3[0],
        }


        

        
        for data in tqdm(test_loader):

            inputs, features, label_list = (
                data['input'].to(DEVICE),
                data['features'].to(DEVICE),
                data['label_list'].to(DEVICE),
                )
            
            target_idx = 0
            label = int(label_list.item())
            patient_id = data['patient_id'][0]
            output = list(model(inputs, features).values())[target_idx]  # NOTE: remember to change the clinical features here!
            pred = torch.sigmoid(output).item()


            # make patient_folder
            patient_folder = os.path.join(output_dir, f"{label}_{patient_id}")
            os.makedirs(patient_folder, exist_ok=True)


            # generate the attributions
            additional_forward_args = (features, True)

            for method in attr_methods_dict.keys():
                if method == "IntegratedGradients":
                    baseline_input_zeros = torch.zeros_like(torch.tensor(inputs))
                    attribution_array = make_integrated_gradients_image(model, target_idx, inputs, additional_forward_args, baseline_input_zeros)

                    # filename of att_map
                    filename = f"Fold{fold_idx}_{pred:.2f}_{method}_zeros.npy"
                    np.save(os.path.join(patient_folder, filename), attribution_array)

                    baseline_input_randn = torch.randn_like(torch.tensor(inputs))
                    attribution_array = make_integrated_gradients_image(model, target_idx, inputs, additional_forward_args, baseline_input_randn)

                    # filename of att_map
                    filename = f"Fold{fold_idx}_{pred:.2f}_{method}_randn.npy"
                    np.save(os.path.join(patient_folder, filename), attribution_array)

                elif method == "Deconvolution":
                    attribution_array = make_deconvolution_image(model, target_idx, inputs, additional_forward_args)

                    # filename of att_map
                    filename = f"Fold{fold_idx}_{pred:.2f}_{method}_standard.npy"
                    np.save(os.path.join(patient_folder, filename), attribution_array)


                elif method == "GradCam++":
                    for layer_name, model_layer in modules_dict_list.items():
                        attribution_array = make_gradcam_image(model, target_idx, inputs, additional_forward_args, model_layer)

                        # filename of att_map
                        filename = f"Fold{fold_idx}_{pred:.2f}_{method}_{layer_name}.npy"
                        np.save(os.path.join(patient_folder, filename), attribution_array)

                elif method == "GradCam":
                    for layer_name, model_layer in modules_dict_list.items():
                        attribution_array = make_gradcam_image(model, target_idx, inputs, additional_forward_args, model_layer, plusplus=False)

                        # filename of att_map
                        filename = f"Fold{fold_idx}_{pred:.2f}_{method}_{layer_name}.npy"
                        np.save(os.path.join(patient_folder, filename), attribution_array)


                elif method == "GuidedGradCam":
                    for layer_name, model_layer in modules_dict_list.items():
                        attribution_array = make_guided_gradcam_image(model, target_idx, inputs, additional_forward_args, model_layer)

                        # filename of att_map
                        filename = f"Fold{fold_idx}_{pred:.2f}_{method}_{layer_name}.npy"
                        np.save(os.path.join(patient_folder, filename), attribution_array)

                else:
                    raise ValueError("Method not found")





if __name__ == '__main__':

    experiment_dir = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\CT_artefact\ResNet10"  # CT_artefact # Xerostomia_M06  # ResNet18
    fold_dir = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\CT_artefact\ResNet10\KFold3"
    filename_model_weights = "DlModel_Weights.pth"
    filename_config = "DlModel_Config.yaml"


    output_dir = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\att_maps\CT_artefact"
    os.makedirs(output_dir, exist_ok=True)

    # select if the config file is loaded with path or global
    config = load_config(os.path.join(fold_dir,"DlModel_Config.yaml")) #get_config('DETOXLung_config')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    # load data
    df_train_val, df_test = load_dataset(config)
    train_transforms, val_transforms = get_transforms(config)
    test_loader, metadata = make_dataloader(config, df_test, val_transforms, validation_mode=True)

    main(experiment_dir, test_loader, metadata, config, output_dir)



    
