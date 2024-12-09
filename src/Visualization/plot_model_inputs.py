import os
import matplotlib.pyplot as plt

from src.visualization.plot_slices import plot_slices



def plot_model_inputs(config, plot_inputs, epoch):
    """
    Plots the model inputs (CT, RTDOSE, Segmentation) used during training, and saves the figure as a png.
    Args:
        config (ConfigObject): configuration object
        plot_inputs (list): list of model input images
        epoch (int): current epoch number
    """
    # check that the directory exists


    save_folder = os.path.join(config['general']['resultsCurrentDirectory'], config['Save']['plot_training_slices']['folder_name'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    n_patients = config['Save']['plot_training_slices']['n_patients_per_epoch']
    for patient_idx in range(min(n_patients, len(plot_inputs))):
        
        ct_range = (config['data']['preprocessing']['ct']['a_max'] - config['data']['preprocessing']['ct']['a_min'])
        rtdose_range = (config['data']['preprocessing']['rtdose']['a_max'] - config['data']['preprocessing']['rtdose']['a_min'])
        
        CT = (plot_inputs[patient_idx][0].cpu() * ct_range) + config['data']['preprocessing']['ct']['a_min']
        RTDOSE = plot_inputs[patient_idx][1].cpu() * rtdose_range + config['data']['preprocessing']['rtdose']['a_min']
        RTSTRUCT = plot_inputs[patient_idx][2].cpu()

        plotting_rows_dicts = [
            {
                "Label": "CT",
                "CT": CT
            },
            {
                "Label": "RTDOSE",
                "RTDOSE": RTDOSE
            },
            {
                "Label": "RTSTRUCT",
                "RTSTRUCT": RTSTRUCT
            }
        ]

        num_CT_slices = CT.shape[0]
        num_plot_slices = config['Save']['plot_training_slices']['n_slices_per_patient']

        slices = list(range(0, num_CT_slices, num_CT_slices // (num_plot_slices + 1)))[1:-1]

        #slices = [20, 30, 40, 50, 60, 70, 80, 90]
        fig, axes = plot_slices(plotting_rows_dicts, slices, RT_region=config['general']['region'])# , title=f"{patient_id} slices")

        filename=os.path.join(save_folder, 'epoch_{}_idx_{}.png'.format(epoch, patient_idx))
        #print(filename)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
