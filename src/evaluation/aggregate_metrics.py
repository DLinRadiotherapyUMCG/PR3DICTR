import os
import pandas as pd
import numpy as np
import logging


from src.utils.saving.create_results_directory import create_results_directory





def aggregate_cross_validation_metrics(config: dict,  k_folds_completed: int, sets: list =['train', 'val']):
    """
    Computes the mean train and validation results for each trial (one trial consists of multiple folds) and saves the results to a csv file.
    
    Args:
        config (dict): config dictionary
        sets (list): list of strings, containing the names of the sets to aggregate (e.g. ['train', 'val', 'test'])
    Returns:
        None
    """

    dfs_list_dict = {set_name: [] for set_name in sets}

    # Loop through the folds, to get the metrics csvs for each fold
    for kFold_index in range(1, k_folds_completed + 1):

        # get the results directory for the current fold
        create_results_directory(config, kFold_index, create_dir = False)

        # Load the results csv file
        for set_name in sets:
            results_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], f"all_{set_name}_metrics.csv")

            df_metrics = pd.read_csv(results_csv_dir, sep=";", index_col=0)
            dfs_list_dict[set_name].append(df_metrics.copy())

    # for each set, calculate the mean and std of the metrics
    folderPath = os.path.join(os.path.join(config["paths"]["results"], config['general']['experiment_name']), config["general"]["trialNumber"])


    endpoint_list = config['columns']['labels']
    for set_name in sets:
        df_set = pd.concat(dfs_list_dict[set_name])

        # Compute the mean and std for each metric and class
        mean_df = df_set.groupby(df_set.index).mean().round(3)
        std_df = df_set.groupby(df_set.index).std().round(3)

        # Merge the mean and std dataframes
        df_out = mean_df.merge(std_df, left_index=True, right_index=True, suffixes=('_mean', '_std'))
        
        # Save the merged dataframe to a CSV file
        
        output_csv_dir = os.path.join(folderPath, f"aggregated_{set_name}_metrics.csv")
        df_out.to_csv(output_csv_dir, sep=";")

        logging.info(f"Saving aggregated {set_name} metrics to {output_csv_dir}")


    





# # # FOR TESTING THIS FUNCTION (DANIEL)

# if __name__ == "__main__":
#     from src.config_presets.tools.get_config import get_config

#     # Load the configuration file
#     config = get_config('Multi_tox')

#     #config['general']['resultsCurrentDirectory'] = "C:/Users/S.P.M. de Vette/OneDrive - UMCG/Desktop/pred_RT_results/Results_0\Trial_2\KFold1"
#     # Call the function
#     aggregate_cross_validation_metrics(config)
    

