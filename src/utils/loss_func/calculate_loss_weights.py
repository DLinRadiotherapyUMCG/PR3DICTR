

def calculate_BCE_loss_weights(config, df_train_val):
    """
    Automatically calculate the pos_weight for the BCE loss function, based on the number of positive and negative samples in the training set.
    Args:
        config (dict): the config params.
        df_train_val (pd.DataFrame): the training and validation dataframe, containing the labels columns
    Returns:
        pos_weight (list): a list of pos_weight values for each endpoint, to be used in the BCE loss function.
    """
    pos_weight = []
    for endpoint in config['columns']['labels']:
        num_pos = (df_train_val[endpoint] == 1).sum()
        num_neg = (df_train_val[endpoint] == 0).sum()
        if num_pos == 0:
            pw = 1.0
        else:
            pw = num_neg / num_pos
        pos_weight.append(float(pw))
    config['training']['loss']['BCE']['pos_weight'] = pos_weight
    return pos_weight