from src.models.dcnn_lrelu import DCNN_LReLU


def get_model(config, dataset_metadata):
    """
    Get the model.
    :param config:
    :return: Model
    """

    model_arguments: dict = config['model']

    n_down_blocks = [sum([x[0] == 2 for x in config['model']['strides']]),
                     sum([x[1] == 2 for x in config['model']['strides']]),
                     sum([x[2] == 2 for x in config['model']['strides']])]
    model_arguments['n_down_blocks'] = n_down_blocks

    if config['model']['model_name'] == 'dcnn_lrelu':
        model_arguments.update(dataset_metadata)
        model = DCNN_LReLU(**model_arguments)
    else:
        raise ValueError(f"Model {config['model']['model_name']} not supported.")

    return model
