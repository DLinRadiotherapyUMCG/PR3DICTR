import matplotlib.colors as mcolors
import colorcet as cc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize



def fade_colormap(cmap_name, fade_length):
    cmap = [mcolors.hex2color(color) for color in cmap_name]
    for i in range(len(cmap)):
        r, g, b = cmap[i]
        a = 0.6
        if i < fade_length:
            exp = pow(i / fade_length, 0.5)
            r, g, b, a = (r * exp, g * exp, b * exp, exp)
        cmap[i] = (r, g, b, a)
    return cmap




# Create a colormap for the negative side of the attention map
neg_cmap = fade_colormap(cc.CET_R4, 0)
neg_cmap = list(reversed(neg_cmap))

# Create a colormap for the positive side of the attention map
pos_cmap = fade_colormap(cc.CET_R4, 0)

# Append positive and negative colormaps to create a single colormap
att_cmap = np.concatenate((neg_cmap, pos_cmap))
# Keep only the positive side for abosulte attention maps
att_cmap_abs = pos_cmap


"""
OLD

------------------------------------------------------------------------------------------------------------------------

NEW
"""

def create_custom_colormap(transparency_threshold=0, red_threshold=0.80):
    # Laad de bestaande cc.CET_R4 colormap
    base_cmap = cc.cm.CET_R4
    
    # Bepaal de start- en eindindex voor de originele colormap
    original_start_index = int(transparency_threshold * 256)
    original_end_index = int(red_threshold * 256)
    
    # Maak een array van de colormap kleuren tussen 0.3 en 0.8
    base_colors = base_cmap(np.linspace(transparency_threshold, red_threshold, original_end_index - original_start_index))
    
    # Voeg transparantie toe voor waarden onder 0.3
    base_colors = np.insert(base_colors, 0, [0, 0, 0, 0], axis=0)
    
    # Voeg donkerrood toe voor waarden boven de rode drempel
    dark_red = np.array([139/255, 0, 0, 1])  # RGBA voor donkerrood
    
    # Maak een array van donkerrode kleuren voor waarden boven de rode drempel
    dark_red_colors = np.tile(dark_red, (256 - original_end_index, 1))  # Herhaal donkerrood voor waarden boven de drempel
    
    # Combineer de kleuren
    colors = np.vstack((base_colors, dark_red_colors))
    colors[243:, 3] *= 0.6
    colors[230:243, 3] *= 0.55
    colors[218:230, 3] *= 0.5
    colors[52:218, 3] *= 0.4
    colors[:52, 3] *= 0.0
    # Converteer kleuren naar lijst van tuples
    color_list = [tuple(color) for color in colors]
    
    return color_list

# Maak de colormap lijst
pos_cmap = create_custom_colormap(0)

# Append positive and negative colormaps to create a single colormap
#att_cmap =  pos_cmap
# Keep only the positive side for abosulte attention maps
att_cmap_abs = pos_cmap