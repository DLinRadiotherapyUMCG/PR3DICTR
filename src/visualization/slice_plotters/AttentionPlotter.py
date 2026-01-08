from matplotlib.colors import LinearSegmentedColormap, Normalize

import src.constants as constants
from .ModalityPlotter import ModalityPlotter



class AttentionPlotter(ModalityPlotter):
    """ Plotting strategy for Attention maps """
    def __init__(self, config, modality_name, global_att_max):
        self.global_att_max = global_att_max

    def set_colourmaps(self):
        plot_min, plot_max = 0, self.global_att_max

        params = constants.PLOTTING_PARAMS["HNC"]["Attention"]
        
        colors_list = params["Attention"]["cmap_abs_colors"]
        N = 256
        self.cmap = LinearSegmentedColormap.from_list("Attention_cmap", colors_list, N=N)

        self.norm = Normalize(vmin=plot_min, vmax=plot_max) 
        self.levels = None


    def plot(self, axs, Attention, slices, params, is_background=False, **kwargs):
        if is_background:
            for i in range(len(axs)):
                axs[i].set_facecolor("black")
        
        alpha = 0.3 # params["alpha"]                                    # TODO: alpha value for attention maps
        for i, slice_n in enumerate(slices):
            axs[i].imshow(Attention[slice_n], cmap=self.cmap, norm=self.norm, alpha=alpha, interpolation='none')


