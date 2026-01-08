import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .slice_plotters.rtdose_colormap import create_RTDOSE_cmap
from .utils.rotate_image_dimensions import rotate_arrs_in_plotting_row_dicts
import src.constants as constants

plt.style.use('default')

# --- Colormap Utilities ---

def create_colormap(cmap_name, min_value, max_value, params=None, N=256):
    if cmap_name == "gray":
        cmap = LinearSegmentedColormap.from_list("CT_cmap", ["black", "white"], N=N)
    elif cmap_name == "RTSTRUCT" or cmap_name == "GTV":
        colors = [(i / N, 0, 0.5) for i in range(N)]  # this colours list is only used if the RTSTRUCT is plotted as a background (so that each struct is a different colour)
        cmap = LinearSegmentedColormap.from_list("RTSTRUCT_cmap", colors, N=N)
    elif cmap_name in ("Attention", "AttentionAbs"):
        colors_list = params["Attention"]["cmap_abs_colors"]
        cmap = LinearSegmentedColormap.from_list("Attention_cmap", colors_list, N=N)
    elif cmap_name == "RTDOSE":
        cmap, norm, _ = create_RTDOSE_cmap(params["RTcmap"])
        return cmap, norm
    elif cmap_name == "PET":
        cmap_name = params["PET"]["cmap"]
        cmap = LinearSegmentedColormap.from_list("PET_cmap", cmap_name, N=N)
    else:
        cmap = mpl.cm.get_cmap(cmap_name, N)

    norm = Normalize(vmin=min_value, vmax=max_value)
    
    return cmap, norm

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
    def plot(self, axs, data, slices, params, **kwargs):
        raise NotImplementedError

class CTPlotter(ModalityPlotter):
    """ Plotting strategy for CT images """
    def plot(self, axs, CT, slices, params, **kwargs):
        # plot the CT image slices along this axis of subplots
        cmap, norm = create_colormap(
            params["cmap"], params["min_val"], params["max_val"], params
        )
        for i, slice_n in enumerate(slices):
            axs[i].imshow(CT[slice_n], cmap=cmap, norm=norm, interpolation='none')

class PETPlotter(ModalityPlotter):
    """ Plotting strategy for PET images """
    def plot(self, axs, PET, slices, params, **kwargs):
        # plot the PET image slices along this axis of subplots
        cmap, norm = create_colormap(params["cmap"], params["min_val"], params["max_val"], params)
        for i, slice_n in enumerate(slices):
            axs[i].imshow(PET[slice_n], cmap=cmap, norm=norm, interpolation='none')

class RTDOSEPlotter(ModalityPlotter):
    """ Plotting strategy for RTDOSE images """
    def plot(self, axs, RTDOSE, slices, params, RTcmap, is_background=False, **kwargs):
        # plot the RTDOSE image slices along this axis of subplots
        cmap, norm, levels = create_RTDOSE_cmap(RTcmap)
        levels = levels[1:]
        for i, slice_n in enumerate(slices):
            slice_data = RTDOSE[slice_n]
            if is_background:
                # plot the RTDOSE as a background (i.e. the colours should not be transparent)
                axs[i].contourf(slice_data, cmap=cmap, norm=norm, levels=levels, alpha=1, origin="upper")
                axs[i].contour(slice_data, levels=levels, norm=norm, colors="white", linewidths=1, alpha=1, origin="upper") # contours are white
            else:
                # plot the RTDOSE with transparency (e.g. for on top of the CT scan)
                axs[i].contourf(slice_data, cmap=cmap, norm=norm, levels=levels, alpha=0.3, origin="lower")
                axs[i].contour(slice_data, cmap=cmap, norm=norm, levels=levels, linewidths=1, origin="lower") # contours are coloured

class RTSTRUCTPlotter(ModalityPlotter):
    """ Plotting strategy for RTSTRUCT images """
    def plot(self, axs, RTSTRUCT, slices, params, is_background=False, **kwargs):
        # plot the RTSTRUCT image slices along this axis of subplots
        cmap, norm = create_colormap(params["cmap"], params["min_val"], params["max_val"], params, N= params["max_val"] + 1)
        for i, slice_n in enumerate(slices):
            slice_data = RTSTRUCT[slice_n]
            if is_background:
                axs[i].imshow(slice_data, cmap=cmap, norm=norm, interpolation='none')
            else:
                axs[i].contour(
                    slice_data > 0,
                    colors=params["color"],
                    linewidths=params["linewidth"],
                    alpha=params["alpha"],
                    origin="lower",
                )

class AttentionPlotter(ModalityPlotter):
    """ Plotting strategy for Attention maps """
    def plot(self, axs, Attention, slices, params, global_att_max, **kwargs):
        # plot the Attention map slices along this axis of subplots
        plot_min, plot_max = 0, global_att_max
        plot_cmap = params["cmap_abs"]
        cmap, norm = create_colormap(plot_cmap, plot_min, plot_max, params)
        alpha = params["alpha"]
        for i, slice_n in enumerate(slices):
            axs[i].imshow(Attention[slice_n], cmap=cmap, norm=norm, alpha=alpha, interpolation='none')

class EmptyPlotter(ModalityPlotter):
    """ Plotting strategy for empty rows (i.e. no data to plot) """
    def plot(self, axs, _, __, params, color="black", **kwargs):
        for ax in axs:
            ax.set_facecolor(color)

# --- Modality Plotter Registry ---

MODALITY_PLOTTERS = {
    "CT": CTPlotter(),
    "RTDOSE": RTDOSEPlotter(),
    "RTSTRUCT": RTSTRUCTPlotter(),
    "GTV" : RTSTRUCTPlotter(),  # GTV uses the same plotting as RTSTRUCT
    "Attention": AttentionPlotter(),
    "PET": PETPlotter(),
    "Empty": EmptyPlotter(),
}

# --- Colorbar Utility ---

def make_row_colorbar(fig, row_ax, num_slices, layer_name, params, global_att_max, RTcmap=None):
    [[x10, y10], [x11, y11]] = row_ax[num_slices - 1].get_position().get_points()
    pad, width, height_scalar = 0.01, 0.01, 1
    max_height = y11 - y10
    left_coord = x11 + pad
    bottom_coord = y10 + (max_height * (1 - height_scalar)) / 2
    cbar_ax = fig.add_axes([left_coord, bottom_coord, width, max_height * height_scalar])
    cbar = None

    if layer_name == "CT":
        cmap, norm = create_colormap(params["CT"]["cmap"], params["CT"]["min_val"], params["CT"]["max_val"], params)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, spacing="proportional")
        cbar.ax.set_title("HU", loc="center", pad=5)
    elif layer_name == "PET":
        cmap, norm = create_colormap(params["PET"]["cmap"], params["PET"]["min_val"], params["PET"]["max_val"], params)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, spacing="proportional")
        cbar.ax.set_title("SUV", loc="center", pad=5)
    elif layer_name == "RTDOSE":
        cmap, norm, levels = create_RTDOSE_cmap(RTcmap)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, spacing="proportional", ticks=levels[1:])
        cbar.ax.set_title("cGy", loc="center", pad=5)
    elif layer_name == "Attention":
        cmap, norm = create_colormap(params["Attention"]["cmap"], -global_att_max, global_att_max, params)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, spacing="proportional")
        cbar.ax.set_title("Attention", loc="center", pad=5)
        cbar.formatter.set_powerlimits((0, 0))
    elif layer_name == "RTSTRUCT" or layer_name == "GTV":
        cmap, norm = create_colormap(params[layer_name]["cmap"], params[layer_name]["min_val"], params[layer_name]["max_val"], params, N=params[layer_name]["max_val"])
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, spacing="proportional")
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
    layer_plotting_order = ["CT", "PET", "RTDOSE", "RTSTRUCT", "GTV", "Attention"]
    colormap_layers = ["Attention", "RTDOSE", "PET", "CT", "GTV", "RTSTRUCT"]

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
        layers_to_plot = [x for x in layer_plotting_order if x in row_dict]
        if verbose:
            print(row_idx, "layers_to_plot:", layers_to_plot)
        colorbar_layer_name = determine_colorbar_layer(layers_to_plot, colormap_layers)

        for layer_idx, layer_name in enumerate(layers_to_plot):
            plotter = MODALITY_PLOTTERS.get(layer_name, MODALITY_PLOTTERS["Empty"])
            params = PLOTTING_PARAMS.get(layer_name, {})
            kwargs = {}
            if layer_name == "RTDOSE":
                kwargs["RTcmap"] = RT_region
                kwargs["is_background"] = (layer_idx == 0 and "Attention" not in layers_to_plot)
                row_dict[layer_name] = rescale_data(
                    row_dict[layer_name],
                    min_val=PLOTTING_PARAMS["RTDOSE"]["min_val"],
                    max_val=PLOTTING_PARAMS["RTDOSE"]["max_val"],
                )
            if layer_name == "RTSTRUCT" or layer_name == "GTV":
                params["min_val"] = 0
                params["max_val"] = row_dict[layer_name].max()
                #print(params["max_val"])
                kwargs["is_background"] = (layer_idx == 0)
            if layer_name == "Attention":
                kwargs["global_att_max"] = global_att_max
                if layer_idx == 0:
                    MODALITY_PLOTTERS["Empty"].plot(axs[row_idx], None, None, params, color=params["background_color"])
            plotter.plot(axs[row_idx], row_dict[layer_name], slice_indexes, params, **kwargs)

        make_row_colorbar(
            fig,
            row_ax=axs[row_idx],
            num_slices=slice_count,
            layer_name=colorbar_layer_name,
            params=PLOTTING_PARAMS,
            RTcmap=RT_region,
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
