import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm, ListedColormap

from .utils.rotate_image_dimensions import rotate_arrs_in_plotting_row_dicts
import src.constants as constants
from .slice_plotters.RTDOSEPlotter import RTDOSEPlotter
from .slice_plotters.RTSTRUCTPlotter import RTSTRUCTPlotter
from .slice_plotters.ImagePlotter import ImagePlotter
from .slice_plotters.AttentionPlotter import AttentionPlotter
from .slice_plotters.DefaultPlotter import DefaultPlotter
from .slice_plotters.EmptyPlotter import EmptyPlotter

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



# --- Colorbar Utility ---

def make_row_colorbar(config, fig, row_ax, num_slices, colorbar_cmap=None, colorbar_norm=None, legend_title = None, global_att_max=None):
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
        cbar.ax.set_title(legend_title, loc="center", pad=5)

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

    print("layer_plotting_order", layer_plotting_order)
    print("colormap_layers", colormap_layers)

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

            # plot images according to their modalitys
            if layer_name in config['data']['preprocessing']['needs_scaling']:
                if layer_name == 'rtdose': # RTDOSE is an image that needs scaling, but according to specific colour thresholds
                    plotter = RTDOSEPlotter(config, layer_name)
                    kwargs["is_background"] = (layer_idx == 0 and "Attention" not in layers_to_plot)
                else:
                    plotter = ImagePlotter(config, layer_name)

            elif layer_name in config['data']['preprocessing']['needs_label_mapping']:
                plotter = RTSTRUCTPlotter(config, layer_name)
                kwargs["is_background"] = (layer_idx == 0)

            elif layer_name == 'attention':
                plotter = AttentionPlotter(config, layer_name, global_att_max)
                
                # if this is the first layer being plotted, make it a background
                kwargs["is_background"] = (layer_idx == 0)
                if layer_idx == 0:
                    # plot a background image
                    EmptyPlotter(config, layer_name).plot(axs[row_idx], None,  params, color=params["background_color"])

            else:
                plotter = DefaultPlotter(config, layer_name)

            params = PLOTTING_PARAMS.get(layer_name, {})
            

            plotter.plot(axs[row_idx], row_dict[layer_name], slice_indexes, params, **kwargs)

            # if this is the layer to use for the colorbar, make it
            if layer_name == colorbar_layer_name:
                colorbar_cmap = plotter.cmap
                colorbar_norm = plotter.norm
                legend_title = plotter.legend_title


        make_row_colorbar(
            config,
            fig,
            row_ax=axs[row_idx],
            num_slices=slice_count,
            colorbar_cmap = colorbar_cmap,
            colorbar_norm = colorbar_norm,
            legend_title = legend_title,
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
