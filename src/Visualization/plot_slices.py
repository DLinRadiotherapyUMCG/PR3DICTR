import matplotlib.colors as mcolors
import colorcet as cc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .rtdose_colormap import create_RTDOSE_cmap

plt.style.use('default')

### DEFINE THE PLOTTING PARAMETERS ###
# the order in which the images are plotted (CT is always bottom, Attention is always top)
layer_plotting_order = ["CT", "RTDOSE", "RTSTRUCT", "Attention"]

# define which layers are to be plotted with a colormap
colormap_layers = [
    "Attention",
    "RTDOSE",
    "CT",
    "RTSTRUCT",
]  # first has most priority, last has least priority


def fade_colormap(cmap_name, fade_length):
    cmap = [mcolors.hex2color(color) for color in cmap_name]
    for i in range(len(cmap)):
        r, g, b = cmap[i]
        a = 1
        if i < fade_length:
            exp = pow(i / fade_length, 0.5)
            r, g, b, a = (r * exp, g * exp, b * exp, exp)
        cmap[i] = (r, g, b, a)
    return cmap


# Create a colormap for the negative side of the attention map
neg_cmap = fade_colormap(cc.CET_L6, 60)
neg_cmap = list(reversed(neg_cmap))

# Create a colormap for the positive side of the attention map
pos_cmap = fade_colormap(cc.CET_L4, 60)

# Append positive and negative colormaps to create a single colormap
att_cmap = np.concatenate((neg_cmap, pos_cmap))
# Keep only the positive side for abosulte attention maps
att_cmap_abs = pos_cmap

HNC_plotting_params = {
    "CT": {
        "cmap": "gray",
        "cmap_title": "HU",
        "min_val": -200,
        "max_val": 400,
    },
    "RTDOSE": {
        "cmap": "dose",
        "cmap_title": "Dose (Gy)",
        "min_val": 0,
        "max_val": 8000,
    },
    "RTSTRUCT": {"color": "deeppink", "linewidth": 2, "alpha": 0.8, "cmap": "gray"},
    "Attention": {
        "cmap": "Attention",
        "cmap_abs": "AttentionAbs",
        "cmap_colors": att_cmap,
        "cmap_abs_colors": att_cmap_abs,
        "cmap_title": None,
        "alpha": 1,
        "background_color": "black",
    },
}


def create_colormap(cmap_name, min_value, max_value, HNC_plotting_params=None, N=256):
    if cmap_name == "gray":
        cmap = LinearSegmentedColormap.from_list("CT_cmap", ["black", "white"], N=N)
    elif cmap_name == "RTSTRUCT":
        if not (min_value == 0 and max_value <= 1):
            N = 16
            # colors = [(0, 0, 0), (0, 1, 0)]
            # cmap = LinearSegmentedColormap.from_list("RTSTRUCT_cmap", colors, N=N)
        # if max_value <= 1:
        colors = [(i / 15, 0, 0.5) for i in range(N)]
        cmap = LinearSegmentedColormap.from_list("RTSTRUCT_cmap", colors, N=N)
    elif cmap_name == "Attention":
        colors_list = HNC_plotting_params["Attention"]["cmap_colors"]
        cmap = LinearSegmentedColormap.from_list("Attention_cmap", colors_list, N=N)
    elif cmap_name == "AttentionAbs":
        colors_list = HNC_plotting_params["Attention"]["cmap_abs_colors"]
        cmap = LinearSegmentedColormap.from_list("Attention_cmap", colors_list, N=N)
    else:
        # Load the colormap from matplotlib
        cmap = mpl.cm.get_cmap(cmap_name, N)

    # Normalize the data to the range -200 to 400
    norm = Normalize(vmin=min_value, vmax=max_value)

    labels = None

    return cmap, norm, labels


def make_row_colorbar(
    fig, row_ax, num_slices, colorbar_layer_name, HNC_plotting_params, global_att_max, RTcmap=None,
):
    # get the coordinates of the last axis in the row, to place the colorbar for this row
    [[x10, y10], [x11, y11]] = (
        row_ax[num_slices - 1].get_position().get_points()
    )  # get coordinates of the last axis
    pad = 0.01
    width = 0.01
    height_scalar = 1  # padding (between img and colorbar) and width for the colorbar
    max_height = y11 - y10  # height of the colorbar
    left_coord = x11 + pad
    bottom_coord = (
        y10 + (max_height * (1 - height_scalar)) / 2
    )  # this ensures the colorbar's height is centered
    # add the colorbar axis
    cbar_ax = fig.add_axes(
        [left_coord, bottom_coord, width, max_height * height_scalar]
    )

    # make a colorbar for the layer that defines the colormap to be used
    cbar = None
    if colorbar_layer_name == "CT":
        cmap, norm, levels = create_colormap(
            cmap_name=HNC_plotting_params["CT"]["cmap"],
            min_value=HNC_plotting_params["CT"]["min_val"],
            max_value=HNC_plotting_params["CT"]["max_val"],
        )
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            spacing="proportional",
        )

    # make the RTDOSE colorbar
    elif colorbar_layer_name == "RTDOSE":
        cmap, norm, levels = create_RTDOSE_cmap(RTcmap)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            spacing="proportional",
            ticks=levels[1:],
        )

    # make the Attention colorbar
    elif colorbar_layer_name == "Attention":
        cmap, norm, levels = create_colormap(
            cmap_name=HNC_plotting_params["Attention"]["cmap"],
            min_value=-global_att_max,
            max_value=global_att_max,
            HNC_plotting_params=HNC_plotting_params,
        )
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            spacing="proportional",
        )
        cbar.formatter.set_powerlimits((0, 0))

    # make the RTSTRUCT colorbar
    elif colorbar_layer_name == "RTSTRUCT":
        cmap, norm, levels = create_colormap(
            cmap_name=HNC_plotting_params["RTSTRUCT"]["cmap"],
            min_value=HNC_plotting_params["RTSTRUCT"]["min_val"],
            max_value=HNC_plotting_params["RTSTRUCT"]["max_val"],
        )
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            spacing="proportional",
        )

    # put a title above the colorbar
    if cbar is not None:
        if colorbar_layer_name == "RTDOSE":
            cbar.ax.set_title("cGy", loc="center", pad=5)
        elif colorbar_layer_name.upper() == "CT":
            cbar.ax.set_title("HU", loc="center", pad=5)
        elif colorbar_layer_name.upper() == "Attention":
            cbar.ax.set_title("Attention", loc="center", pad=5)

        cbar.ax.yaxis.set_ticks_position("right")


""" Data processing utilities"""


def rescale_data(data, min_val, max_val):
    """
    Rescales the data from the [0,1] range (to the CT or RTDoSE ranges).
    """
    if data.min() >= 0 and data.max() <= 1:
        return data * (max_val - min_val) + min_val
    else:
        return data


def get_slice_dimensions(input_dict):
    """
    Finds the dimensions of the images in the input dict, used to define the size of the axes.
    """
    for img_type in layer_plotting_order:
        if img_type in input_dict.keys():
            img = input_dict[img_type]
            return img.shape

    raise ValueError("No images found in input_dict")


def determine_colorbar_layer(layers_to_plot, colormap_layers):
    """
    Finds the layer that should be used to create the colorbar, based on the layers that are to be plotted.
    Args:
        layers_to_plot (list): list of layers to be plotted
        colormap_layers (list): list of layers that can be plotted with a colormap. The order of this list defines the priority with which the colorbar is made.
    Returns:
        lyr (string: the name of the layer that should be used to create the colorbar
    """
    for lyr in colormap_layers:
        if lyr in layers_to_plot:
            return lyr


""" Plotting functions. Differentiated by the type of image to be plotted. """


def plot_CT(axs, CT, slices, HNC_plotting_params):
    """
    Plots the CT slices using the same CT colormap for all slices.
    """
    cmap, norm, levels = create_colormap(
        cmap_name=HNC_plotting_params["CT"]["cmap"],
        min_value=HNC_plotting_params["CT"]["min_val"],
        max_value=HNC_plotting_params["CT"]["max_val"],
    )

    for i, slice_n in enumerate(slices):
        axs[i].imshow(CT[slice_n], cmap = "bone")#, cmap=cmap, norm=norm)


def plot_RTDOSE(axs, RTDOSE, slices, RTcmap, is_background=False):
    """
    Plots the RTDOSE slcies using the colormap specified by RTcmap.
    """

    # retrieve the colormap for RTDOSE (HNC)
    cmap, norm, levels = create_RTDOSE_cmap(RTcmap)
    levels = levels[1:]  # remove the first level, as it is the background level

    for i, slice_n in enumerate(slices):
        RTDOSE_slice = RTDOSE[slice_n]

        if is_background:
            contourf = axs[i].contourf(
                RTDOSE_slice,
                cmap=cmap,
                norm=norm,
                levels=levels,
                alpha=1,
                origin="upper",
            )
            contour = axs[i].contour(
                RTDOSE_slice,
                levels=levels,
                norm=norm,
                colors="white",
                linewidths=1,
                alpha=1,
                origin="upper",
            )

            # # Display the image with imshow
            # print(contourf.get_array())
            # axs[i].imshow(contourf.get_array()
            # axs[i].imshow(RTDOSE_slice, cmap=cmap, norm=norm, levels=levels, alpha=1, origin="lower")  # NOTE: see if its possible to stick the contour under the attention layer
            # contourf.remove()
            # axs[i].imshow(contour.get_array())
            # contour.remove()
        else:
            contourf = axs[i].contourf(
                RTDOSE_slice,
                cmap=cmap,
                norm=norm,
                levels=levels,
                alpha=0.3,
                origin="lower",
            )
            contour = axs[i].contour(
                RTDOSE_slice,
                cmap=cmap,
                norm=norm,
                levels=levels,
                linewidths=1,
                origin="lower",
            )

            # # Display the image with imshow
            # print(contourf.get_array())
            # axs[i].imshow(contourf.get_array())
            # contourf.remove()
            # axs[i].imshow(contour.get_array())
            # contour.remove()


def plot_RTSTRUCT(axs, RTSTRUCT, slices, HNC_plotting_params, is_background=False):
    """
    Plots the RTSTRUCT image on the axes provided. If the RTSTRUCT is the background of the plot, then the contours are not plotted and imshow() is used instead.
    """
    cmap, norm, levels = create_colormap(
        cmap_name=HNC_plotting_params["RTSTRUCT"]["cmap"],
        min_value=HNC_plotting_params["RTSTRUCT"]["min_val"],
        max_value=HNC_plotting_params["RTSTRUCT"]["max_val"],
    )
    for i, slice_n in enumerate(slices):
        RTSTRUCT_slice = RTSTRUCT[slice_n]

        if is_background:
            # can just use imshow if the RTSTRUCT is the background (no contours needed for the RTSTRUCT)
            axs[i].imshow(RTSTRUCT_slice, cmap=cmap, norm=norm)
        else:
            # plots just the contours of the RTSTRUCT
            axs[i].contour(
                RTSTRUCT_slice > 0,
                colors="black",
                linewidths=HNC_plotting_params["RTSTRUCT"]["linewidth"] + 4,
                alpha=HNC_plotting_params["RTSTRUCT"]["alpha"] * 0.25,
                origin="lower",
            )  # ensures orientation is correct (up/down)
            axs[i].contour(
                RTSTRUCT_slice > 0,
                colors=HNC_plotting_params["RTSTRUCT"]["color"],
                linewidths=HNC_plotting_params["RTSTRUCT"]["linewidth"],
                alpha=HNC_plotting_params["RTSTRUCT"]["alpha"],
                origin="lower",
            )  # ensures orientation is correct (up/down)


def plot_Attention(axs, Attention, slices, HNC_plotting_params, global_att_max):
    """
    Plots the attention map slices on the axes provided.
    """
    # Set min and max to the largest absolute value in the attention map
    # This centers the colormap around 0, and ensures that the colormap is symmetric
    att_min = np.min(Attention)
    # If there is negative attention, set the colormap to be symmetric around 0, and
    # pick a symmetric colormap
    if att_min < 0:
        plot_min = -global_att_max
        plot_max = global_att_max
        plot_cmap = HNC_plotting_params["Attention"]["cmap"]
    # If there is no negative attention, set the colormap to be from 0 to the largest
    # value and pick a colormap that is not symmetric
    else:
        plot_min = 0
        plot_max = global_att_max
        plot_cmap = HNC_plotting_params["Attention"]["cmap_abs"]

    cmap, norm, levels = create_colormap(
        cmap_name=plot_cmap,
        min_value=plot_min,
        max_value=plot_max,
        HNC_plotting_params=HNC_plotting_params,
    )
    alpha = HNC_plotting_params["Attention"]["alpha"]
    for i, slice_n in enumerate(slices):
        axs[i].imshow(Attention[slice_n], cmap=cmap, norm=norm, alpha=alpha)


def plot_empty_img(axs, color="black"):
    """
    Plots an empty image on the axes provided.
    """
    for ax in axs:
        ax.set_facecolor(color)
    return axs


"""  """


"""  """


""" MAIN """


def plot_slices(
    row_dicts,
    slice_indexes,
    title=None,
    HNC_plotting_params=HNC_plotting_params,
    RTcmap="HNC",
    verbose=False,
):
    slice_count = len(slice_indexes)
    row_count = len(row_dicts)

    # Calculate the size of the figure based on the size of the plots
    if "Attention" not in row_dicts[0]:

        im_shape = get_slice_dimensions(row_dicts[0])
    else:
        im_shape = row_dicts[0]["Attention"][0].shape

    plot_width = im_shape[-2]
    plot_height = im_shape[-1]

    fig_width = plot_width * slice_count / 30
    fig_height = plot_height * row_count / 28

    # Create a gridspec to add a colorbar axis
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        row_count,
        slice_count,
        width_ratios=[1] * slice_count,
        hspace=0,
        wspace=0.1,
        figure=fig,
    )

    # Initialize global max and min values for the attention maps
    global_att_max = float('-inf')

    # Loop over all rows and calculate min and max values
    for row_dict in row_dicts:
        if "Attention" in row_dict:
            ATTENTION = row_dict["Attention"]
            local_min = np.min(ATTENTION)
            local_max = np.max(ATTENTION)
            abs_max = max(abs(local_min), abs(local_max))
            global_att_max = max(global_att_max, abs_max)

    # Create title
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.95)

    # Create subplots
    axs = [
        [plt.subplot(gs[i, j]) for j in range(slice_count)] for i in range(row_count)
    ]

    # loop over the plot rows
    for row_idx in range(row_count):
        row_dict = row_dicts[row_idx]
        # loop through the dictionary keys and plot the images in the correct order (e.g. CT first)
        layers_to_plot = [x for x in layer_plotting_order if x in row_dict.keys()]
        if verbose:
            print(row_idx, "layers_to_plot:", layers_to_plot)
        # reversed_layers_to_plot = layers_to_plot[::-1]

        # figure out which layer the colormap should come from
        colorbar_layer_name = determine_colorbar_layer(layers_to_plot, colormap_layers)

        im = None
        for layer_idx, layer_name in enumerate(layers_to_plot):
            if layer_name == "CT":
                CT = row_dict["CT"]
                #CT = rescale_data(
                #    CT,
                #    min_val=HNC_plotting_params["CT"]["min_val"],
                #    max_val=HNC_plotting_params["CT"]["max_val"],
                #)

                plot_CT(axs[row_idx], CT, slice_indexes, HNC_plotting_params)

            elif layer_name == "RTDOSE":
                RTDOSE = row_dict["RTDOSE"]
                RTDOSE = rescale_data(
                    RTDOSE,
                    min_val=HNC_plotting_params["RTDOSE"]["min_val"],
                    max_val=HNC_plotting_params["RTDOSE"]["max_val"],
                )
                rtdose_is_background = (
                    True
                    if layers_to_plot[0] == "RTDOSE"
                    and "Attention" not in layers_to_plot
                    else False
                )
                plot_RTDOSE(
                    axs[row_idx],
                    RTDOSE,
                    slice_indexes,
                    RTcmap,
                    is_background=rtdose_is_background,
                )

            elif layer_name == "Attention":
                if layer_idx == 0:
                    plot_empty_img(
                        axs[row_idx],
                        color=HNC_plotting_params["Attention"]["background_color"],
                    )

                ATTENTION = row_dict["Attention"]

                plot_Attention(
                    axs[row_idx], ATTENTION, slice_indexes, HNC_plotting_params, global_att_max
                )

            elif layer_name == "RTSTRUCT":
                RTSTRUCT = row_dict["RTSTRUCT"]
                HNC_plotting_params["RTSTRUCT"]["min_val"] = 0
                HNC_plotting_params["RTSTRUCT"]["max_val"] = RTSTRUCT.max()

                rtstruct_is_background = True if layer_idx == 0 else False
                plot_RTSTRUCT(
                    axs[row_idx],
                    RTSTRUCT,
                    slice_indexes,
                    HNC_plotting_params,
                    is_background=rtstruct_is_background,
                )

        make_row_colorbar(
            fig,
            row_ax=axs[row_idx],
            num_slices=slice_count,
            colorbar_layer_name=colorbar_layer_name,
            HNC_plotting_params=HNC_plotting_params,
            RTcmap=RTcmap,
            global_att_max=global_att_max,
        )

    # Only show axes at the outer edges
    for i in range(row_count):
        for j in range(slice_count):
            axs[i][j].set_aspect("equal")
            if i != row_count - 1:
                axs[i][j].xaxis.set_visible(False)
            # else:
            #    axs[i][j].xaxis.set_ticks([0,100])  # NOTE: can correct the ticks here!
            if j != 0:
                axs[i][j].yaxis.set_visible(False)
            # else:
            #    axs[i][j].yaxis.set_ticks([0,100])

    # Add titles to each row
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
