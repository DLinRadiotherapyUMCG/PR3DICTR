import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .rtdose_colormap import create_RTDOSE_cmap
from .utils.rotate_image_dimensions import rotate_arrs_in_plotting_row_dicts
import src.constants as constants

plt.style.use('default')

# --- Colormap Utilities ---

def rescale_data(data, min_val, max_val):
    if data.min() >= 0 and data.max() <= 1:
        return data * (max_val - min_val) + min_val
    return data

def get_slice_dimensions(input_dict, layer_order):
    for img_type in layer_order:
        if img_type in input_dict:
            img = input_dict[img_type]
            return img.shape
    raise ValueError("No images found in input_dict")

# --- Plotting Strategy Classes ---

class ModalityPlotter:
    """ Base class for modality plotting strategies """
    def __init__(self, config, modality_name):

        self.modality_config = config['data']['preprocessing'][modality_name]
        self.modality_name = modality_name
        self.RT_region = config['general']['region']

        self.set_colourmaps()

    def set_colourmaps(self):
        pass

    def plot(self, config, axs, data, slices, params, **kwargs):
        raise NotImplementedError


class DefaultPlotter(ModalityPlotter):
    """ Default plotting strategy (just a placeholder)"""
    def plot(self, config, axs, data, slices, modality_name, params, **kwargs):

        for i, slice_n in enumerate(slices):
            axs[i].imshow(data[slice_n], cmap='viridis', interpolation='none')


class ImagePlotter(ModalityPlotter):
    """ Plotting strategy for images (CT, PET, MR, etc.)"""

    def set_colourmaps(self, N = 256):
        cmap_name = self.modality_config["cmap"]
        min_val = self.modality_config["a_min"]
        max_val = self.modality_config["a_max"]

        self.cmap = mpl.cm.get_cmap(cmap_name, N)
        self.norm  = Normalize(vmin=min_val, vmax=max_val)
        self.levels = None

    def plot(self, config, axs, image_array, slices, modality_name, params, **kwargs):
        # plot the CT image slices along this axis of subplots

        for i, slice_n in enumerate(slices):
            axs[i].imshow(image_array[slice_n], cmap=self.cmap, norm=self.norm, interpolation='none')



# config, axs[row_idx], row_dict[layer_name], slice_indexes, layer_name, params,

class RTDOSEPlotter(ModalityPlotter):
    """ Plotting strategy for RTDOSE images """


    def set_colourmaps(self, N = 256):
        #RT_region = self.RT_region
        self.cmap, self.norm, self.levels = create_RTDOSE_cmap(self.RT_region)
        self.levels = self.levels[1:]


    def plot(self, config, axs, RTDOSE, slices, modality_name, params, is_background=False, **kwargs):
        # plot the RTDOSE image slices along this axis of subplots
        cmap = self.cmap
        norm = self.norm
        levels = self.levels

        for i, slice_n in enumerate(slices):
            slice_data = RTDOSE[slice_n]
            if is_background:
                # plot the RTDOSE as a background (i.e. the colours should not be transparent)
                axs[i].imshow(slice_data, cmap=cmap, norm=norm) # , interpolation='none')
                # axs[i].contourf(slice_data, cmap=cmap, norm=norm, levels=levels, alpha=1, origin="upper")
                axs[i].contour(slice_data, levels=levels, norm=norm, colors="white", linewidths=1, alpha=1) # , origin="upper") # contours are white
            else:
                # plot the RTDOSE with transparency (e.g. for on top of the CT scan)
                axs[i].contourf(slice_data, cmap=cmap, norm=norm, levels=levels, alpha=0.3) # , origin="lower")
                axs[i].contour(slice_data, cmap=cmap, norm=norm, levels=levels, linewidths=1) # , origin="lower") # contours are coloured


class RTSTRUCTPlotter(ModalityPlotter):
    """ Plotting strategy for RTSTRUCT images """

    def set_colourmaps(self):
        min_val = 0
        max_val = len(self.modality_config['target_labels']) - 1
        N = max_val + 1
        #modality_config = self.modality_config
        colors = [(i / N, 0, 0.5) for i in range(N)]  # this colours list is only used if the RTSTRUCT is plotted as a background (so that each struct is a different colour)
        
        cmap_name = self.modality_config["cmap"]
        self.cmap = mpl.cm.get_cmap(cmap_name, N)
        #self.cmap = LinearSegmentedColormap.from_list("RTSTRUCT_cmap", colors, N=N)
        self.norm = Normalize(vmin=min_val, vmax=max_val)
        self.levels = None


    def plot(self, config, axs, RTSTRUCT, slices, modality_name, params, is_background=False, **kwargs):
        # plot the RTSTRUCT image slices along this axis of subplots
        for i, slice_n in enumerate(slices):
            slice_data = RTSTRUCT[slice_n]
            if is_background:
                axs[i].imshow(slice_data, cmap= self.cmap, norm=self.norm, interpolation='none')
            else:
                axs[i].contour(
                    slice_data > 0,
                    colors=self.modality_config["outline_color"],
                    linewidths=self.modality_config["linewidth"],
                    alpha=self.modality_config["alpha"],
                    origin="lower",
                )

class AttentionPlotter(ModalityPlotter):
    """ Plotting strategy for Attention maps """
    def set_attention_colourmaps(self, global_att_max):
        plot_min, plot_max = 0, global_att_max

        params = constants.PLOTTING_PARAMS["HNC"]["Attention"]
        
        colors_list = params["Attention"]["cmap_abs_colors"]
        N = 256
        self.cmap = LinearSegmentedColormap.from_list("Attention_cmap", colors_list, N=N)

        self.norm = Normalize(vmin=plot_min, vmax=plot_max) 
        self.levels = None


    def plot(self, config, axs, Attention, slices, params, global_att_max, is_background=False, **kwargs):
        if is_background:
            for i in range(len(axs)):
                axs[i].set_facecolor("black")
        
        alpha = params["alpha"]
        for i, slice_n in enumerate(slices):
            axs[i].imshow(Attention[slice_n], cmap=self.cmap, norm=self.norm, alpha=alpha, interpolation='none')



class EmptyPlotter(ModalityPlotter):
    """ Plotting strategy for empty rows (i.e. no data to plot) """
    def set_colourmaps(self):
        self.cmap = None
        self.norm = None

    def plot(self, config, axs, _, __, params, color="black", **kwargs):
        for ax in axs:
            ax.set_facecolor(color)

# --- Modality Plotter Registry ---

MODALITY_PLOTTERS = {
    # "rtdose": RTDOSEPlotter(),
    # "rtstruct": RTSTRUCTPlotter(),
    # 'segmentation_map': RTSTRUCTPlotter(),  # RTSTRUCT uses the same plotting as segmentation_map
    # "gtv" : RTSTRUCTPlotter(),  # GTV uses the same plotting as RTSTRUCT
    "attention": AttentionPlotter,
    "empty": EmptyPlotter,

    #"image": ImagePlotter(),
}

# --- Colorbar Utility ---

def make_row_colorbar(config, fig, row_ax, num_slices, colorbar_cmap=None, colorbar_norm=None, global_att_max=None):
    [[x10, y10], [x11, y11]] = row_ax[num_slices - 1].get_position().get_points()
    pad, width, height_scalar = 0.01, 0.01, 1
    max_height = y11 - y10
    left_coord = x11 + pad
    bottom_coord = y10 + (max_height * (1 - height_scalar)) / 2
    cbar_ax = fig.add_axes([left_coord, bottom_coord, width, max_height * height_scalar])

    if colorbar_cmap is None or colorbar_norm is None:
        cbar = None
    else:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=colorbar_norm, cmap=colorbar_cmap), cax=cbar_ax, spacing="proportional")

    
    if cbar is not None:
        cbar.ax.yaxis.set_ticks_position("right")

# --- Main Plotting Function ---

def get_plotting_params(RT_region):
    try:
        return constants.PLOTTING_PARAMS[RT_region]
    except KeyError:
        raise ValueError(f"Unknown RT region: {RT_region}")


def determine_colorbar_layer(layers_to_plot, colormap_layers):
    # function to figure out which plotted layer should be used for the colorbar (i.e. the most important one)
    for lyr in colormap_layers:
        if lyr in layers_to_plot:
            return lyr


def plot_slices(
        config,
        row_dicts,
        slice_indexes,
        title=None,
        RT_region="HNC",
        plotting_axis="axial",
        verbose=False,
    ):
    """
    Plots multiple rows of image slices with different modalities.
    Args:
        row_dicts (list of dict): each dict contains the different image modalities to plot for that row
        slice_indexes (list of int): list of slice indexes to plot for each row 
        title (str, optional): title for the entire figure
        RT_region (str): region for which the RTDOSE colormap should be configured
        plotting_axis (str): axis along which to plot the slices ('axial', 'sagittal', 'coronal')
        verbose (bool): whether to print verbose output
    Returns:
        fig: matplotlib figure object
        axs: list of axes in the figure
    """
    
    PLOTTING_PARAMS = get_plotting_params(RT_region)
    layer_plotting_order = ["CT", "PET", "RTDOSE", "RTSTRUCT", "segmentation_map", "GTV", "Attention"]
    colormap_layers = ["Attention", "RTDOSE", "PET", "CT", "GTV", "RTSTRUCT", "segmentation_map"]

    layer_plotting_order = [l.lower() for l in layer_plotting_order]
    colormap_layers = [l.lower() for l in colormap_layers]

    slice_count = len(slice_indexes)
    row_count = len(row_dicts)

    im_shape = get_slice_dimensions(row_dicts[0], layer_plotting_order)
    plot_width, plot_height = im_shape[-2], im_shape[-1]
    fig_width = plot_width * slice_count / 30
    fig_height = plot_height * row_count / 28

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(row_count, slice_count, width_ratios=[1] * slice_count, hspace=0, wspace=0.1, figure=fig)

    # Find global attention max
    global_att_max = max(
        max(abs(np.min(row.get("Attention", [0]))), abs(np.max(row.get("Attention", [0]))))
        for row in row_dicts if "Attention" in row
    ) if any("Attention" in row for row in row_dicts) else 1

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.95)

    axs = [[plt.subplot(gs[i, j]) for j in range(slice_count)] for i in range(row_count)]

    if plotting_axis.lower() != "axial":
        row_dicts = rotate_arrs_in_plotting_row_dicts(row_dicts, layer_plotting_order, plotting_axis)

    for row_idx, row_dict in enumerate(row_dicts):
        layers_to_plot = [x.lower() for x in layer_plotting_order if x in row_dict]
        if verbose:
            print(row_idx, "layers_to_plot:", layers_to_plot)
        colorbar_layer_name = determine_colorbar_layer(layers_to_plot, colormap_layers)

        for layer_idx, layer_name in enumerate(layers_to_plot):
            kwargs = {}


            if layer_name in config['data']['preprocessing']['needs_scaling']:
                if layer_name == 'rtdose':
                    plotter = RTDOSEPlotter(config, layer_name)
                    kwargs["is_background"] = (layer_idx == 0 and "Attention" not in layers_to_plot)

                else:
                    plotter = ImagePlotter(config, layer_name)

            elif layer_name in config['data']['preprocessing']['needs_label_mapping']:
                plotter = RTSTRUCTPlotter(config, layer_name)
                kwargs["is_background"] = (layer_idx == 0)
                    
            # elif layer_name in MODALITY_PLOTTERS.keys():
            #     # print(layer_name, MODALITY_PLOTTERS)
            #     plotter = MODALITY_PLOTTERS.get(layer_name, MODALITY_PLOTTERS[layer_name])
            elif layer_name == 'attention':
                plotter = AttentionPlotter(config, layer_name, global_att_max=global_att_max)
                plotter.set_attention_colourmaps(self, global_att_max)
            else:
                plotter = DefaultPlotter(config, layer_name)

            params = PLOTTING_PARAMS.get(layer_name, {})
            
            if layer_name == "attention":
                kwargs["global_att_max"] = global_att_max
                if layer_idx == 0:
                    # plot a background image
                    EmptyPlotter(config, layer_name).plot(config, axs[row_idx], None, None, params, color=params["background_color"])
                    # MODALITY_PLOTTERS["empty"].plot(config, axs[row_idx], None, None, params, color=params["background_color"])

            plotter.plot(config, axs[row_idx], row_dict[layer_name], slice_indexes, layer_name, params, **kwargs)


            # if this is the layer to use for the colorbar, make it
            if layer_name == colorbar_layer_name:
                colorbar_cmap = plotter.cmap
                colorbar_norm = plotter.norm


        make_row_colorbar(
            config,
            fig,
            row_ax=axs[row_idx],
            num_slices=slice_count,
            colorbar_cmap = colorbar_cmap,
            colorbar_norm = colorbar_norm,
            global_att_max=global_att_max,
        )

    for i in range(row_count):
        for j in range(slice_count):
            axs[i][j].set_aspect("equal")
            if i != row_count - 1:
                axs[i][j].xaxis.set_visible(False)
            if j != 0:
                axs[i][j].yaxis.set_visible(False)

    for i, ax in enumerate(axs):
        ax[0].annotate(
            row_dicts[i]["Label"],
            xy=(0, 0.5),
            xytext=(-ax[0].yaxis.labelpad - 5, 0),
            xycoords=ax[0].yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    return fig, axs
