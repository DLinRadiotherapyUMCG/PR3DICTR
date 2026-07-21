import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm, ListedColormap
import copy 

from .ModalityPlotter import ModalityPlotter    


class RTSTRUCTPlotter(ModalityPlotter):
    """ Plotting strategy for RTSTRUCT images """

    def set_colourmaps(self):
        min_val = 0
        max_val = len(self.modality_config['target_labels']) 
        N = max_val 
        cmap_source = self.modality_config["cmap"]
        
        # Handle both matplotlib colormap names and lists of color names
        if isinstance(cmap_source, list):
            # Replace empty strings with transparent color
            self.empty_indices = [i for i, color in enumerate(cmap_source) if color == '']
            processed_colors = [(0, 0, 0, 0) if color == "" else color for color in cmap_source]
            self.cmap = LinearSegmentedColormap.from_list("RTSTRUCT_cmap", processed_colors, N=N)
        else:
            self.empty_indices = []
            self.cmap = mpl.colormaps[cmap_source].resampled(N)

        self.norm = Normalize(vmin=min_val, vmax=max_val)
        levels = np.arange(min_val, max_val + 1)
        self.norm = BoundaryNorm(boundaries=np.arange(min_val, max_val+1), ncolors=len(levels))

        if isinstance(cmap_source, list):
            colors_list = [self.cmap(self.norm(i)) for i in range(0, max_val)]
        else:
            colors_list = ['black'] + [self.cmap(self.norm(i)) for i in range(1, max_val)]

        self.cmap_w_background = ListedColormap(colors_list)
        self.norm_w_background = BoundaryNorm(boundaries=np.arange(0, max_val+1), ncolors=len(colors_list))

    def set_single_outline_colour(self, is_background=False):

        #color=
        # print(self.modality_config)
        self.empty_indices = []  # reset the empty indices
        self.cmap = ListedColormap([self.modality_config["outline_color"]])
        self.norm = Normalize(vmin=0, vmax=1)
        self.levels = [0.5, 1.5]

        if is_background:
            self.cmap_w_background = mpl.colormaps[self.modality_config["normalised_cmap"]].resampled(256)
            self.norm_w_background = Normalize(vmin=0, vmax=1)

            self.cmap = self.cmap_w_background
            self.norm = self.norm_w_background


    def plot(self, axs, RTSTRUCT, slices, params, is_background=False, **kwargs):

        if RTSTRUCT.max() <= 1:
            self.set_single_outline_colour(is_background=is_background)

        # plot the RTSTRUCT image slices along this axis of subplots
        for i, slice_n in enumerate(slices):
            slice_data = copy.deepcopy(RTSTRUCT[slice_n])
            slice_data[np.isin(slice_data, self.empty_indices)] = 0  # set empty indices to background (0)
            if is_background:
                axs[i].imshow(slice_data, cmap=self.cmap_w_background, norm=self.norm_w_background, interpolation='none')
                #axs[i].imshow(slice_data, cmap= self.cmap, norm=self.norm, interpolation='none')
            else:
                for label in range(1, slice_data.max()+1):
                    mask = slice_data == label
                    if np.any(mask):
                        axs[i].contour(mask, levels=[0.5], colors=[self.cmap(self.norm(label))], linewidths=2, alpha=1)

            del slice_data