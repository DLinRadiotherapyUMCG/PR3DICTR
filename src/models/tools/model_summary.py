from torchinfo import summary
from src.utils.fileHandler import create_folder, create_textfile


def get_model_summary(config, model, input_size, device, save_to_file = True):
    
    """
    Get model summary and number of trainable parameters. Saves the summary to a text file, within the Kfold folder, if specified.

    Args:
        config (dict): configuration dictionary
        model (torch.nn.Module): the model to summarize
        input_size (tuple or list of tuples): input size for the model summary
        device (torch.device or string): device to use for the model summary
        save_to_file (bool): whether to save the summary to a text file

    Returns:
        total_params (int): number of trainable parameters

    """
    # Get and save summary
    verbose = 0
    txt = str(summary(model=model, input_size=input_size, device=device, verbose=verbose, depth=5,
                      col_names=["input_size", "output_size", "num_params", "kernel_size"], row_settings=["var_names"]))
    if save_to_file:
        create_textfile(config['general']['resultsCurrentDirectory'], config['saving']['filenames']['model_summary'], txt)

    # Determine number of trainable parameters
    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    # total_params = sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params