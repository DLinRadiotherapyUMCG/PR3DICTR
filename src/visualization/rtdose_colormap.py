import matplotlib as mpl
import numpy as np

import src.constants as constants




""" Functions to turn a color mapping dictionary into a matplotlib colormap """


def remove_duplicate_RGB_values(dictionary):
    """
    Removes keys from the dictionary that have the same value as the key value under it.
    i.e. if 4000 and 4500 have the same colour, remove 4500
    Args:
        dictionary (dict): dictionary with keys and values
    """
    values = set()
    keys_to_remove = []
    for key, value in dictionary.items():
        if value in values:
            keys_to_remove.append(key)
        else:
            values.add(value)
    for key in keys_to_remove:
        del dictionary[key]
    
    return dictionary




def create_RTDOSE_cmap(name):
    """ Create a custom colormap from a dictionary
    
    Args: 
        colors <dict>: Levels to colornames
    Returns:
        cmap <matplotlib.colors.ListedColormap>: Custom colormap
        norm <matplotlib.colors.BoundaryNorm>: Normalize the colorbar
        labels <list>: Labels for the colorbar
    """
    try:
        colormapping_dict = constants.PLOT_SLICES_COLOURMAPS[name]
    except KeyError:
        raise ValueError(f"Unknown colormap name: {name}")
    
    assert isinstance(colormapping_dict, dict)

    colormapping_dict = dict(sorted(colormapping_dict.items(), key=lambda x: x[0]))

    colormapping_dict = remove_duplicate_RGB_values(colormapping_dict)

    
    labels = sorted(colormapping_dict.keys())
    clrs = [colormapping_dict[l] for l in labels[:-1]]
    clrs = np.array(clrs)/255.
    levels = labels
    cmap, norm = mpl.colors.from_levels_and_colors(levels, clrs)
    return cmap, norm, labels
