import os
import matplotlib.pyplot as plt

from src.visualization.plot_slices import plot_slices



def adjust_modality(config, image, modality):
    """
    Adjusts the image to the correct range based on the modality.
    Args:
        image (torch.tensor): image to adjust
        modality (str): modality of the image
    Returns:
        torch.tensor: adjusted image
    """

    if modality == 'CT':
        ct_range = (config['data']['preprocessing']['ct']['a_max'] - config['data']['preprocessing']['ct']['a_min'])
        image = (image * ct_range) + config['data']['preprocessing']['ct']['a_min']

    elif modality == 'RTDOSE':
        rtdose_range = (config['data']['preprocessing']['rtdose']['a_max'] - config['data']['preprocessing']['rtdose']['a_min'])
        image = image * rtdose_range + config['data']['preprocessing']['rtdose']['a_min']

    elif modality == 'PET':
        pet_range = (config['data']['preprocessing']['pet']['a_max'] - config['data']['preprocessing']['pet']['a_min'])
        image = (image * pet_range) + config['data']['preprocessing']['pet']['a_min']

    elif modality == 'RTSTRUCT' or modality == 'GTV':
        image = image

    return image




def plot_model_inputs(config, plot_inputs, epoch):
    """

    Plots the model inputs (CT, RTDOSE, Segmentation) used during training, and saves the figure as a png.
    Args:
        config (dict): configuration object
        plot_inputs (list): list of model input images
        epoch (int): current epoch number
    Returns:
        None

    """
    # check that the directory exists


    save_folder = os.path.join(config['general']['resultsCurrentDirectory'], config['saving']['plot_training_slices']['folder_name'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    n_patients = config['saving']['plot_training_slices']['n_patients_per_epoch']
    for patient_idx in range(min(n_patients, len(plot_inputs))):

        # list of dicts, where each dict is one image type to plot (i.e. one row)
        plotting_rows_dicts = []

        num_slices = plot_inputs[patient_idx][0].shape[0]
        num_plot_slices = config['saving']['plot_training_slices']['n_slices_per_patient']
        slices = list(range(0, num_slices, num_slices // (num_plot_slices + 1)))[1:-1]


        for image_channel_idx, image_modality_name in enumerate(config['data']['image_keys']):

            image_modality_name = image_modality_name.upper()

            if image_modality_name == "SEGMENTATION_MAP":
                image_modality_name = "RTSTRUCT"
            
            # get the image out of the tensor
            image = plot_inputs[patient_idx][image_channel_idx].cpu()
            # adjust the scale of the image
            scaled_image = adjust_modality(config, image, image_modality_name)

            row_dict = {"Label" : image_modality_name,
                        f"{image_modality_name}": scaled_image}
            plotting_rows_dicts.append(row_dict)


        fig, axes = plot_slices(plotting_rows_dicts, slices, RT_region=config['general']['region'])# , title=f"{patient_id} slices")
        
        filename=os.path.join(save_folder, 'epoch_{}_idx_{}.png'.format(epoch, patient_idx))
        fig.savefig(filename, bbox_inches='tight')

        plt.close(fig)
        plt.clf()