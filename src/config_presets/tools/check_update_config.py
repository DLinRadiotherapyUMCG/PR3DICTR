




def update_config(base, updates):
    """
    Recursively updates the base dictionary with values from the updates dictionary.
    Only non-dictionary values are overwritten. \
    (i.e. if only one thing in the 'model' key needs to be updated, the rest of the 'model' dict will remain the same)
    """
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            update_config(base[key], value)
        else:
            base[key] = value