from src.constants import MAIN_DATA_SOURCES
import os



def alter_filename_if_external_dataset(config, filename : str):
    """
    If the data source is not the main data source (i.e. it is an external test set), 
    then alter the filename to include the data source name

    Args:
        config (dict): config dictionary
        filename (str): filename to alter (if needed)
    Returns:
        filename (str): altered filename
    """
    data_source = config['data']['source']
    if data_source not in MAIN_DATA_SOURCES:
        filename, file_extension = os.path.splitext(filename)  
        filename = f"{filename}_{data_source}{file_extension}"  # e.g. predictions_MDACC.csv

    return filename