import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import misc 
import config
from sklearn.calibration import calibration_curve



def make_calibration_plot(config, predictions_dict, true_labels_dict, set_name, filename=None):
    """
    Makes a calivration plot using a set predicitons and true labels. The predictions are split into 10 bins for this.
    The plot is then saved to a file if a `filename` is provided.
    Args:
        config: a ConfigObject.
        predictions_dict: a dictionary of lists of predictions for each endpoint.
        true_labels_dict: a dictionary of lists of true labels for each endpoint.
        set_name: a string, the name of the set (e.g. 'train', 'val', 'test') to be used in the title of the plot.
        filename: a string, the name of the file to save the plot to.
    Returns:
        None
    """
    #colours = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:cyan']
    colours = make_colorlist(config)

    x = [0, 1]
    y = [0, 1]

    plt.figure(figsize=(6, 6))
    plt.plot([0,1], [0,1], 'black', label="Ideal") # plot the ideal calibration line
    
    for idx, endpoint in enumerate(config.endpoint_list):
        y_pred = predictions_dict[endpoint].cpu().numpy()
        y_true = true_labels_dict[endpoint].cpu().numpy()

        mask = [True if x in config.valid_endpoint_values else False for x in y_true]

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, pos_label=1, n_bins=10, strategy="quantile")

        coef = np.polyfit(mean_predicted_value, fraction_of_positives, 1)
        linear_fit = np.poly1d(coef)

        # plot the points (scatter)
        plt.plot(mean_predicted_value, fraction_of_positives, 'o', color=colours[idx])
        # plot the best fit line
        #plt.plot(x, linear_fit(y), '--', label=endpoint,  color=colours[idx])
        plt.plot(mean_predicted_value, fraction_of_positives, '-', label=endpoint,  color=colours[idx])

        #plt.plot(mean_predicted_value, fraction_of_positives, 'o-', color=colours[idx])
    
    plt.title("Calibration Plot: {}".format(set_name))
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted Rate")
    plt.ylabel("Observed Rate")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.close()



def main():
    dir = os.path.join("experiments", "HP_ResNet_LayerFreezing")

    #print(os.listdir(dir))
    trial_folders = [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]

    for t_id in trial_folders:
        #print()

        trial_fold_folders = [x for x in os.listdir(os.path.join(dir, t_id)) if os.path.isdir(os.path.join(dir, t_id, x))]
        #trial_folders = [x for x in trial_folders if x.split("_")[2] in []]
        for fold_folder in trial_fold_folders:
            try:
                trial_folder_path = os.path.join(dir, t_id, fold_folder)
                #print(os.listdir(path))
                files = os.listdir(trial_folder_path)
                all_outputs_csv = [x for x in files if "all_outputs.csv" in x][0]
                #print(s.path.join(trial_folder_path, all_outputs_csv))
                df_outputs = pd.read_csv(os.path.join(trial_folder_path, all_outputs_csv), delimiter=';') # load predictions csv
                df_outputs = df_outputs[df_outputs['Mode'] == "val"] # keep only the validation data
                #print(len(df_outputs))

                predictions_dict = {}
                true_labels_dict = {}
                for endpoint in config.endpoint_list:
                    predictions_dict[endpoint] = df_outputs[endpoint + "_pred"].to_numpy()
                    true_labels_dict[endpoint] = df_outputs[endpoint + "_true"].to_numpy()
                cal_plot_path = os.path.join(trial_folder_path, 'calibration_plot_validation.png')
                make_calibration_plot(config, predictions_dict, true_labels_dict, set_name="Validation", filename=cal_plot_path)
                
            except Exception as e:
                #print("EXCEPTION:", e)
                pass


if __name__ == "__main__":
    main()