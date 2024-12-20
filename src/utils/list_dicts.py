


def append_to_list_dicts(config, list_dicts = [], value_dicts = []):
    """
    Takes a list of list_dicts and a list of value_dicts and appends the values of the value_dicts to the list_dicts.
    """
    assert len(list_dicts) == len(value_dicts)

    endpoint_list = config['columns']['labels']

    for idx in range(len(list_dicts)):
        for endpoint in endpoint_list:
            
            list_dicts[idx][endpoint].append(value_dicts[idx][endpoint])
    
    return list_dicts