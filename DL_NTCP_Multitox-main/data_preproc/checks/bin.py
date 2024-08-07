






def plot_multiple_arrs(arr_list, arr_names_list, overlay_list, overlay_names_list, alpha_overlay_list, nr_images, figsize, cmap_list, cmap_overlay_list,
                       colorbar_title_list, filename, vmin_list, vmin_overlay_list, vmax_list, vmax_overlay_list,
                       ticks_steps_list, segmentation_map_list):
    """
    Plot slices of multiple arrays. Each Numpy on a different row, e.g. CT (row 1), RTDOSE (row 2) and
    segmentation_map (row 3).

    Args:
        arr_list (list): list of Numpy arrays with slices to be plotted, each Numpy array with shape
            (z, y, x) = (num_slices, num_rows, num_columns)
        overlay_list (list): list of Numpy arrays with slices to be plotted overlayed on arr_list.
            Note: len(arr_list) == len(overlay_list). If overlay_list[i] = None, then no overlay.
        alpha_overlay_list (float): overlay alpha, in [0, 1].
        nr_images (int): number of slices to plot for each input
        figsize (tuple): size of output figure
        cmap_list (list): list of cmaps
        cmap_overlay_list (list): list of cmaps for arrays in overlay_list
        colorbar_title_list (list): list of titles of colorbar
        filename (str): save location
        vmin_list (list): list minimum value for colormap
        vmin_overlay_list (list): list minimum value for arrays in overlay_list
        vmax_list (list): list minimum value for colormap
        vmax_overlay_list (list): list minimum value for colormap for arrays in overlay_list
        ticks_steps_list (list): list of steps of value display for colormap
        segmentation_map_list (list): list of segmentation_maps for drawing the contours of the structures

    Returns:

    """
    assert len(arr_list) == len(overlay_list)

    # Make sure that every input array has the same number of slices
    nr_slices = arr_list[0].shape[0]
    for i in range(1, len(arr_list)):
        assert nr_slices == arr_list[i].shape[0]
    # List of structures for which we may plot contours (i.e. if segmentation_map is not None)
    structures = data_prc_cfg.draw_contour_structures

    # Initialize variables
    if nr_images is None:
        nr_images = nr_slices

    # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
    if nr_slices < nr_images:
        nr_images = nr_slices
    slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)

    # Only consider unique values
    slice_indices = np.unique(slice_indices.astype(int))

    # Determine number of columns and rows
    num_cols = nr_images
    num_rows = len(arr_list)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=tuple(figsize))

    for row, (arr, arr_overlay) in enumerate(zip(arr_list, overlay_list)):
        
        arr_name = arr_names_list[row]

        # Make sure that arr and arr_overlay shapes are identical
        if arr_overlay is not None:
            assert arr.shape == arr_overlay.shape
            overlay_name = overlay_names_list[row]
            alpha_overlay = alpha_overlay_list[row]
            cmap_overlay = cmap_overlay_list[row]
            vmin_overlay = vmin_overlay_list[row] if vmin_overlay_list[row] is not None else arr_overlay.min()
            vmax_overlay = vmax_overlay_list[row] if vmax_overlay_list[row] is not None else arr_overlay.max()
            #if overlay_name == "RTDOSE":
            #    overlay_ticks = overlay_ticks_steps_list[row]

        cmap = cmap_list[row]
        colorbar_title = colorbar_title_list[row]

        # Colormap
        # Data range
        vmin = vmin_list[row] if vmin_list[row] is not None else arr.min()
        vmax = vmax_list[row] if vmax_list[row] is not None else arr.max()

        # Label ticks
        # '+1' because end of interval is not included
        #ticks = np.arange(vmin, vmax + 1, ticks_steps_list[row]) if ticks_steps_list[row] is not None else None
        ticks_steps = ticks_steps_list[row]
        if arr_name == "RTDOSE":
            ticks = ticks_steps
        else:
            ticks = np.arange(vmin, vmax+1, ticks_steps) if (ticks_steps is not None) else None

        #ticks_steps = 
        #ticks = np.arange(vmin, vmax+1, ticks_steps) if ticks_steps is isinstance(ticks_steps_list[row], (int, float)) else ticks_steps

        for i, idx in enumerate(slice_indices):
            # Consider the first and last slice
            if i == 0:
                idx = 0
            if i == nr_images - 1:
                idx = nr_slices - 1

            ax[row, i].set_aspect('equal')
            idx = int(idx)
            if arr_name == "RTDOSE":
                plot_RTDOSE(ax[row, i], arr[idx, ...], cmap, data_prc_cfg.rtdose_norm, labels=ticks, alpha=1)
                ax[row, i].set_xticks([])
                ax[row, i].set_yticks([])
            else:
                im = ax[row, i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)
                ax[row, i].axis('off')

            if arr_overlay is not None:
                if overlay_name == "RTDOSE":
                    RTDOSE_slice = arr_overlay[idx, ...]
                    RTDOSE_cmap = cmap_overlay
                    norm = data_prc_cfg.rtdose_norm
                    contour_value_thresholds = HNC_contour_value_thresholds
                    ax[row, i].contourf(RTDOSE_slice, cmap=RTDOSE_cmap, norm=norm, levels=contour_value_thresholds, alpha=alpha_overlay)
                    ax[row, i].contour(RTDOSE_slice, cmap=RTDOSE_cmap, norm=norm, levels=contour_value_thresholds, linewidths=1, alpha=1)
                    #plot_RTDOSE(ax[row, i], arr_overlay[idx, ...], cmap_overlay, data_prc_cfg.rtdose_norm, labels=ticks, alpha=1, overlay=True)
                    ax[row, i].set_xticks([])
                    ax[row, i].set_yticks([])
                else:
                    im = ax[row, i].imshow(arr_overlay[idx, ...], alpha=alpha_overlay, cmap=cmap_overlay, vmin=vmin_overlay,
                                         vmax=vmax_overlay)

            if segmentation_map_list[row] is not None:
                for struct in structures:
                    struct_value = data_prc_cfg.structures_values[struct]
                    ax[row, i].contour(segmentation_map_list[row][idx, ...] == struct_value,
                                       colors=data_prc_cfg.draw_contour_color,
                                       linewidths=data_prc_cfg.draw_contour_linewidth)
            

        #plt.tight_layout()

        # Add colorbar
        if colorbar_title is not None:
            fig.subplots_adjust(right=0.8)
            max_height = 0.925
            min_height = 1 - max_height
            length = max_height - min_height
            length_per_input = length / num_rows
            epsilon = 0.05
            bottom = max_height - (row + 1) * length_per_input + epsilon / 2
            cbar = fig.add_axes(rect=[0.825, bottom, 0.01, length_per_input - epsilon])
            cbar.set_title(colorbar_title)

            if arr_name == "RTDOSE":
                fig.colorbar(mpl.cm.ScalarMappable(norm= data_prc_cfg.rtdose_norm, cmap=cmap), cax=cbar, ticks=ticks, spacing='proportional')

                plt.rcParams["axes.edgecolor"] = "0.15"
                plt.rcParams["axes.linewidth"]  = 1.25
            else:
                fig.colorbar(im, cax=cbar, ticks=ticks, spacing='proportional')

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

