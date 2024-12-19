import os

from src.utils.saving.alter_filename_for_external_dataset import alter_filename_if_external_dataset

def get_predictions_csv_dir(config, test_set=False, ensemble_predictions=False, external_set=False):
    

    if ensemble_predictions:
        output_filename = config['Save']['filenames']['ensemble_predictions_csv']
    elif test_set:
        output_filename = config['Save']['filenames']['test_set_predictions_csv']
    else:
        output_filename = config['Save']['filenames']['predictions_csv']

    # if using an external data source (e.g. for validating the model), then alter the predictions filename
    output_filename = alter_filename_if_external_dataset(config, output_filename)

    output_file_dir = os.path.join(config['general']['resultsCurrentDirectory'], output_filename)

    return output_file_dir


# predictions_csv_{}.csv



"""

 filenames:
    model_summary: "model_summary.txt"
    config_yaml: "DlModel_Config.yaml"
    model_weights: "DlModel_Weights.pth"
    predictions_csv: "model_predictions.csv"
    test_set_predictions_csv: "test_set_predictions.csv"
    ensemble_predictions_csv: "ensemble_predictions.csv"
      
    

"""