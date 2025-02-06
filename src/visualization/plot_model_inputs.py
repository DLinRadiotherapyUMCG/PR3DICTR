import os
import matplotlib.pyplot as plt

from src.visualization.plot_slices import plot_slices

def SelectModality(config, ptn, index):
        dictFound = None
        modalityKeys = config['data']['image_keys']
        imageKey = modalityKeys[index]
        if("ct" in imageKey):
            ct_range = (config['data']['preprocessing']['ct']['a_max'] - config['data']['preprocessing']['ct']['a_min'])
            dictFound = {"Label": "CT",
                         "Name": "CT",
                         "CT": ((ptn[index].cpu()* ct_range) + config['data']['preprocessing']['ct']['a_min'])}
        elif("rtdose"):
            rtdose_range = (config['data']['preprocessing']['rtdose']['a_max'] - config['data']['preprocessing']['rtdose']['a_min'])
            dictFound = {"Label": "RTDOSE",
                         "Name": "RTDOSE",
                         "RTDOSE": (ptn[index].cpu()* rtdose_range + config['data']['preprocessing']['rtdose']['a_min'])}
        elif("mri"):
            #TODO handle MRI images
             dictFound = {"Label": "MRI",
                         "Name": "MRI",
                         "MRI": ptn[index].cpu()}
        elif("segmentation_map"):
            dictFound = {"Label": "RTSTRUCT",
                         "Name": "segmentation_map",
                         "RTSTRUCT": ptn[index].cpu()}
        else:
            # Otherwise also a structure
            dictFound = {"Label": "RTSTRUCT",
                         "Name": imageKey,
                         "RTSTRUCT": ptn[index].cpu()}
        return dictFound

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
        
        # Check the image modalities used from config
        modalityKeys = config['data']['image_keys']
        plotting_rows_dicts = []
        ptn = plot_inputs[patient_idx]
        for i in range(len(modalityKeys)):
            dictImage = SelectModality(config, ptn, i)
            plotting_rows_dicts.append(dictImage)

        # Expect always 1 image modality available and get the amount of slices
        num_CT_slices = ptn[0].shape[0]
        num_plot_slices = config['saving']['plot_training_slices']['n_slices_per_patient']

        slices = list(range(0, num_CT_slices, num_CT_slices // (num_plot_slices + 1)))[1:-1]

        #slices = [20, 30, 40, 50, 60, 70, 80, 90]
        fig, axes = plot_slices(plotting_rows_dicts, slices, RT_region=config['general']['region'])# , title=f"{patient_id} slices")

        filename=os.path.join(save_folder, 'epoch_{}_idx_{}.png'.format(epoch, patient_idx))
        #print(filename)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
            



