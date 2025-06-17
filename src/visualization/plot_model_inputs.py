import os
import matplotlib.pyplot as plt

from src.visualization.plot_slices import plot_slices



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

        slices = list(range(0, num_CT_slices, num_CT_slices // (num_plot_slices + 1)))[1:-1]

        #slices = [20, 30, 40, 50, 60, 70, 80, 90]
        fig, axes = plot_slices(plotting_rows_dicts, slices, RT_region=config['general']['region'])# , title=f"{patient_id} slices")

        filename=os.path.join(save_folder, 'epoch_{}_idx_{}.png'.format(epoch, patient_idx))
        #print(filename)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
            



