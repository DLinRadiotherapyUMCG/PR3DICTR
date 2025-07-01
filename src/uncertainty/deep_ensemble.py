import torch

from src.dataset.load_dataset import load_dataset, generate_K_fold_cross_validation_splits
from src.models.tools.get_classification_model import get_classification_model
from src.dataset.get_dataloader import make_dataloader   
from src.uncertainty.utils import set_dataset_config_for_uncertainty
from src.utils.set_random_seed import set_random_seed
from src.dataset.get_transforms import get_transforms
from src.utils.loss_func.get_loss_function import get_loss_function
from src.evaluation.mainMetricHandler import mainMetricHandler




def get_dataset_for_uncertainty_experiments(config):
    # Load the dataset and dataloader
    # load the data, and make K-fold splits
    df_train_val, df_test = load_dataset(config)
    k_fold_dataframes_list = generate_K_fold_cross_validation_splits(config, df_train_val)
    train_transforms, val_transforms = get_transforms(config)

    dataset_split_dict = k_fold_dataframes_list[0]  # the first split is used for training and validation
    train_data, val_data = dataset_split_dict['train'], dataset_split_dict['val']

    train_loader, metadata = make_dataloader(config, train_data, train_transforms, validation_mode=False)
    val_loader, _ = make_dataloader(config, val_data, val_transforms, validation_mode=True)
    test_loader, _ = make_dataloader(config, df_test, val_transforms, validation_mode=True)

    return train_loader, val_loader, test_loader, metadata



def train_deep_ensemble_model(config):
    """
    Train a deep ensemble model.
    """
    
    set_random_seed(config['general']['seed'])  # Set the random seed for reproducibility

    # override the dataloader settings
    set_dataset_config_for_uncertainty(config)

    
    # Load the dataset and dataloader
    # load the data, and make K-fold splits
    train_loader, val_loader, test_loader, metadata = get_dataset_for_uncertainty_experiments(config)
    
    # get the loss function and metric handler
    loss_function = get_loss_function(config)
    metricHandler = mainMetricHandler(config)


    print("DATASETS LOADED")

    # 1. load the dataset and dataloader
            # may have to use an uncertainty config to overwrite the dataloader settings in the original config
    # 2. Initialize the model

    pass  # Replace with actual implementation


