import matplotlib as mpl
import numpy as np



""" The dictionary of colour mapping for HNC RTDOSE  """
# keys: dose value thresholds
# values: RGB colour tuples

HNC_color_mapping_dict = {
    8400: (64,0,0),
    8050: (128,0,0),
    7700: (128,0,0),
    7490: (255,0,0),
    7000: (255,128,0),
    6650: (255,128,0),
    6300: (255,255,0),
    5805: (0,64,0),
    5425: (0,255,0),
    5154: (0,255,0),
    5000: (0,255,255),
    4883: (0,255,255),
    4000: (0,128,192),
    3000: (0,0,160),
    2000: (0,0,160),
    1000: (192,192,192),
    500: (192,192,192),
    200: (255,255,255),
    0: (255,255,255),
}

LUNG_color_mapping_dict = {
    6900: (248,0,33),
    6700: (248,0,33),
    6420: (255,118,74),
    6000: (46,255,13),
    5700: (46,255,13),
    5400: (66,255,253),
    5000: (251,255,35),
    4000: (27,62,126),
    3000: (5,0,253),
    2000: (5,0,253),
    1000: (245,245,245),
    500: (255,255,255),
    200: (255,255,255),
}




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
    if name == "HNC":
        colormapping_dict = HNC_color_mapping_dict
    elif name == "LUNG":
        colormapping_dict = LUNG_color_mapping_dict
    else:
        raise ValueError(f"Unknown colormap name: {name}")
    
    assert(isinstance(colormapping_dict, dict))

    colormapping_dict = dict(sorted(colormapping_dict.items(), key=lambda x: x[0]))

    colormapping_dict = remove_duplicate_RGB_values(colormapping_dict)

    
    labels = sorted(colormapping_dict.keys())
    clrs = [colormapping_dict[l] for l in labels[:-1]]
    clrs = np.array(clrs)/255.
    levels = labels
    cmap, norm = mpl.colors.from_levels_and_colors(levels, clrs)
    return cmap, norm, labels
