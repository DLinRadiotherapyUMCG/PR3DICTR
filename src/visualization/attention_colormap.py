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

