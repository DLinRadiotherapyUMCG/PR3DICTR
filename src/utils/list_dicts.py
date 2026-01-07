
def append_to_list_dicts(config : dict, list_dicts : list = [], value_dicts : list = []):
    """
    Takes a list of list_dicts and a list of value_dicts and appends the values of the value_dicts to the list_dicts.
    Args:
        config (dict): the configuration dict
        list_dicts (list): a list of dictionaries
        value_dicts (list): a list of dictionaries
    Returns:
        list_dicts (list): a list of dictionaries
    """
    assert len(list_dicts) == len(value_dicts)

    endpoint_list = config['columns']['labels']

    for idx in range(len(list_dicts)):
        for endpoint in endpoint_list:
            
            list_dicts[idx][endpoint].append(value_dicts[idx][endpoint])
    
    return list_dicts