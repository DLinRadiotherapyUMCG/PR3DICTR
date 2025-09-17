import os
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq, CommentedMap



def save_config(config):
    """
    This function will make the config file that can be saved after excecuting the 
    main function.
    """
    
    current_dir = config['general']['resultsCurrentDirectory']
    out_name_config = config['saving']['filenames']['config_yaml']

    out_dir = os.path.join(current_dir, out_name_config)

    config = format_config_lists_to_flow_style(config) # convert lists to flow style for better readability in yaml
    
    print(f"Saving config to {out_dir}")
    print("DANIEL")

    yaml = YAML()
    yaml.width = 120

    
    with open(out_dir, "w") as f:
        yaml.dump(config, f)

    return




    
def format_config_lists_to_flow_style(obj):
    """
    Recursively converts all lists (including ruamel.yaml's CommentedSeq) within a nested data structure
    to flow style (i.e., [a, b, c]) for YAML serialization, while preserving inline and end-of-line comments.
    This is useful when you want lists to be represented in flow style in the output YAML file,
    regardless of their original style or how they were created.
    
    Args:
      obj: The input object (the config), which can be a ruamel.yaml CommentedMap, CommentedSeq, plain list,
              or any nested combination thereof.
    Returns:
      obj: The same data structure, but with all lists/CommentedSeqs set to flow
    
    Examples
    >>> from ruamel.yaml import YAML
    >>> from ruamel.yaml.comments import CommentedMap, CommentedSeq
    >>> data = CommentedMap({
    ...     'numbers': [1, 2, 3],
    ...     'nested': CommentedMap({
    ...         'letters': ['a', 'b', 'c']
    ...     })
    ... })
    >>> lists_to_flow_style(data)
    # When dumped with ruamel.yaml, 'numbers' and 'letters' will appear as [1, 2, 3] and [a, b, c] in YAML.
    >>> # Handles plain lists as well:
    >>> lists_to_flow_style({'plain_list': [4, 5, 6]})
    # Output YAML will show: plain_list: [4, 5, 6]
    """
    
    if isinstance(obj, CommentedMap):
        for k, v in obj.items():
            obj[k] = format_config_lists_to_flow_style(v)
    elif isinstance(obj, CommentedSeq):
        for i, v in enumerate(obj):
            obj[i] = format_config_lists_to_flow_style(v)
        obj.fa.set_flow_style()   # enable [a, b, c] style
    elif isinstance(obj, list):  # plain list (e.g. created programmatically)
        seq = CommentedSeq([format_config_lists_to_flow_style(v) for v in obj])
        seq.fa.set_flow_style()
        return seq
    return obj