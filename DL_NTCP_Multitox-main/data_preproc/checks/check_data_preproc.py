"""
This script:
    - Creates and saves plot for the outputs arrays of data_preproc.py.
    - For each patient and each structure: for segmentation_map: calculate percentage of voxels that are at CT's voxel,
        where CTs voxel values are less than or equal to some value (e.g. -200 HU). This is to check which structure
        lies outside the actual human body. This could indicate incorrect segmentation.

Useful note:
    - plt.imshow(): the dimensions (M, N) of input array define the rows and columns of the image, respectively.
        I.e. in the figures of an array with shape (z, y, x) = (z, M, N) = (num_slices, num_rows, num_columns):
        - z-axis: scan slice. Starting with slice 0 (i.e. bottom human body part in the scan), until last slice
            (upper human body part in the scan). Subplots in the figure are created from left to right, top to bottom.
        - y-axis (rows): moves up and down of the image.
        - x-axis (columns): moves left and right of the image.
    - main_segmentation_outside_ct() with "ct < max_outside_lower_limit" does not work for slices with metal artifacts.
        That is, the metal artifacts changes the HU values of inside CT. As a result  many voxels will be considered
        "outside" the CT, but are actually inside the CT.

    Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    - The very first and very last slices are always included in the subplot.
    - The pixel values of the RTDOSE subplots cannot be compared between RTDOSE subplots, because the subplots are
        created individually and so the pixel values are normalized differently for each subplot.
"""
import sys
sys.path.append('./data_preproc')
sys.path.append('./data_preproc/checks/')
import os
import time
import math
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# Source: https://github.com/matplotlib/mplfinance/issues/386
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import itertools
from multiprocessing import Pool

import data_preproc_config as data_prc_cfg
from data_preproc_functions import create_folder_if_not_exists, Logger, sort_human
#from rtdose_colormaps import contour_value_thresholds as HNC_contour_value_thresholds
from plotting_functions import plot_slices, create_RTDOSE_cmap





def plot_RTDOSE(ax, arr, cmap, norm, labels, alpha, overlay=False):
    #ax.imshow(np.ones_like(arr), cmap='grey')
    if not overlay:
        arr = np.flipud(arr)
    ax.contourf(arr, cmap=cmap, norm=norm, levels=labels, alpha=alpha)
    
    if overlay:
        _,_,levels = create_RTDOSE_cmap(name="HNC")
        ax.contour(arr, cmap=cmap, norm=norm, levels=levels, linewidths=0.5, alpha=1)
    else:
        ax.contour(arr, colors="white", norm=norm, levels=labels[1:], linewidths=0.5, alpha=1)

    

def plot_arr(arr, nr_images, figsize, cmap, colorbar_title, filename, vmin=None, vmax=None, ticks_steps=None,
             segmentation_map=None, RTdose=False):
    """
    Plot and save slices of array. The slices that we consider are equal distant from each other in the whole array.
    For example, if the array has 100 slices, and nr_images = 4, then the slices with the following index
    will be plotted and saved: [0, 33, 66, 99].
    The first and last plot will always be the first and last slice, respectively.

    If nr_images is None, then consider all slices!

    Result of determining the number of columns and rows:
        nr_slices: 1
            num_cols: 1
            num_rows: 1
        nr_slices: 2
            num_cols: 1
            num_rows: 2
        nr_slices: 3
            num_cols: 2
            num_rows: 2
        nr_slices: 4
            num_cols: 2
            num_rows: 2
        nr_slices: 5
            num_cols: 2
            num_rows: 3
        ...

    Args:
        arr (Numpy array): array with slices to be plotted, of shape (z, y, x) = (num_slices, num_rows, num_columns)
        nr_images (int): number of slices to plot
        figsize (tuple): size of output figure
        cmap (str): colormap, see https://matplotlib.org/stable/tutorials/colors/colormaps.html
        colorbar_title (str): title of colorbar
        filename (str): save location
        vmin (int/float): (optional) minimum value for colormap: smaller values will be clipped to this minimum value
        vmax (int/float): (optional) maximum value for colormap: larger values will be clipped to this maximum value
        ticks_steps (int): (optional) steps of value display for colormap
        segmentation_map (Numpy array): (optional) segmentation_map for drawing the contours of the structures

    Returns:

    """
    # Initialize variables
    nr_slices = arr.shape[0]
    if nr_images is None:
        nr_images = nr_slices
    # List of structures for which we may plot contours (i.e. if segmentation_map is not None)
    structures = data_prc_cfg.draw_contour_structures

    # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
    if nr_slices < nr_images:
        nr_images = nr_slices
    slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)

    # Only consider unique values
    slice_indices = np.unique(slice_indices.astype(int))

    # Estimate the number of columns and rows
    num_cols_rows = len(slice_indices) ** (1 / 2)
    # Determine first decimal of square-root: this is important in determining the exact number of columns and rows
    # num_cols_rows = 5.50001 results in num_cols_rows_without_rounding = 5.5
    # num_cols_rows = 5.49999 results in num_cols_rows_without_rounding = 5.4
    num_cols_rows_without_rounding = math.floor(num_cols_rows * 10) / 10
    base_num_cols_rows = int(num_cols_rows_without_rounding)
    decimal_num_cols_rows = num_cols_rows_without_rounding - base_num_cols_rows
    if decimal_num_cols_rows == 0:
        num_cols = num_rows = base_num_cols_rows
    elif decimal_num_cols_rows < 0.5:
        num_cols = base_num_cols_rows
        num_rows = base_num_cols_rows + 1
    else:
        num_cols = base_num_cols_rows + 1
        num_rows = base_num_cols_rows + 1

    # # Converting to log()-values is useful for large value differences, such as in RTDOSE
    # # Map arr from [a, b] to [1, b-a+1] (e.g. from [-1000, 1000] to [1, 2001]) before applying log()
    # arr = (arr - arr.min()) + 1
    # arr = np.log(arr)

    # Colormap
    # Data range
    vmin = vmin if vmin is not None else arr.min()
    vmax = vmax if vmax is not None else arr.max()
    # Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    # Label ticks
    # '+1' because end of interval is not included
    if RTdose:
        ticks = ticks_steps
    else:
        ticks = np.arange(vmin, vmax+1, ticks_steps) if (ticks_steps is not None) else None
    

    # fig = plt.figure(figsize=tuple(figsize))
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=tuple(figsize))
    ax = ax.flatten()

    for i, idx in enumerate(slice_indices):
        # Consider the first and last slice
        if i == 0:
            idx = 0
        if i == nr_images - 1:
            idx = nr_slices - 1

        idx = int(idx)
        ax[i].set_aspect('equal')
        if RTdose:
            plot_RTDOSE(ax[i], arr[idx, ...], cmap, data_prc_cfg.rtdose_norm, labels=ticks, alpha=1)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            #ax[i].label_outer()
        else:
            im = ax[i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[i].axis('off')


        if segmentation_map is not None:
            for struct in structures:
                struct_value = data_prc_cfg.structures_values[struct]
                ax[i].contour(segmentation_map[idx, ...] == struct_value,
                              colors=data_prc_cfg.draw_contour_color,
                              linewidths=data_prc_cfg.draw_contour_linewidth)
            

    plt.tight_layout()

    # Add colorbar
    fig.subplots_adjust(right=0.8)
    # add_axes([xmin, ymin, dx, dy])
    cbar = fig.add_axes([0.825, 0.15, 0.02, 0.7])
    cbar.set_title(colorbar_title)
    if RTdose:
        fig.colorbar(mpl.cm.ScalarMappable(norm= data_prc_cfg.rtdose_norm, cmap=cmap), cax=cbar, ticks=ticks, spacing='proportional')
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"]  = 1.25

    else:
        fig.colorbar(im, cax=cbar, ticks=ticks, spacing='proportional')

    plt.savefig(filename)
    plt.close(fig)
    # Source: https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib











def main_segmentation_outside_ct():
    # Initialize variables
    use_umcg = data_prc_cfg.use_umcg
    mode_list = data_prc_cfg.mode_list
    data_dir_mode_list = data_prc_cfg.save_dir_mode_list
    patient_id_length = data_prc_cfg.patient_id_length

    save_root_dir = data_prc_cfg.save_root_dir
    save_dir_dataset_full = data_prc_cfg.save_dir_dataset_full
    max_outside_fraction = data_prc_cfg.max_outside_fraction
    max_outside_lower_limit = data_prc_cfg.max_outside_lower_limit
    logger = Logger(os.path.join(save_root_dir, data_prc_cfg.filename_data_preproc_ct_segmentation_map_logging_txt))
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, data_prc_cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get patient_ids
        patients_list = os.listdir(os.path.join(save_dir_dataset_full))
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # TODO: temporary: consider all patients, for exclude_patients.csv
        # Testing for a small number of patients
        if data_prc_cfg.test_patients_list is not None:
            test_patients_list = data_prc_cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        # Make sure that the order and length is equal
        # patients_list = [x.replace('.npy', '') for x in patients_npy_list]
        # assert patients_npy_list == [x + '.npy' for x in patients_list]

        df = pd.DataFrame()
        print(patients_list)
        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {id}'.format(id=patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Load CT and segmentation
            # arr = np.load(os.path.join(save_dir_dataset_full, patient_id_npy))
            ct = np.load(os.path.join(save_dir_dataset_full, patient_id, data_prc_cfg.filename_ct_npy))
            segmentation_map = np.load(os.path.join(save_dir_dataset_full, patient_id, data_prc_cfg.filename_segmentation_map_npy))

            assert ct.shape == segmentation_map.shape

            outside_dict = dict()
            for structure in data_prc_cfg.parotis_structures + data_prc_cfg.submand_structures:
                # Total segmentation_map
                mask = (segmentation_map == data_prc_cfg.structures_values[structure])
                total_nr_voxels = mask.sum()

                # Segmentation_map outside CT of human body
                mask_outside = (ct < max_outside_lower_limit) * mask
                outside_nr_voxels = mask_outside.sum()

                # Determine fraction of segmentation outside CT of human body
                fraction_outside = outside_nr_voxels / total_nr_voxels
                if fraction_outside > max_outside_fraction:
                    logger.my_print('Patient_id = {}, structure = {}, fraction outside CT = {}'
                                    .format(patient_id, structure, fraction_outside),
                                    level='warning')
                outside_dict[structure] = fraction_outside

            # Save descriptive statistics
            df_i = pd.DataFrame(outside_dict, index=[patient_id])
            df = pd.concat([df, df_i], axis=0)

        # Save as csv file
        df = df.round(data_prc_cfg.nr_of_decimals)
        df.to_csv(os.path.join(save_root_dir, data_prc_cfg.filename_segmentation_fraction_outside_ct_csv), sep=';')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')
    logger.close()
    del logger


def main_plot():
    # Initialize variables
    use_umcg = data_prc_cfg.use_umcg
    mode_list = data_prc_cfg.mode_list
    data_dir_mode_list = data_prc_cfg.save_dir_mode_list
    patient_id_length = data_prc_cfg.patient_id_length

    save_root_dir = data_prc_cfg.save_root_dir
    figures_dir = data_prc_cfg.save_dir_figures
    figures_all_dir = data_prc_cfg.save_dir_figures_all
    logger = Logger(os.path.join(save_root_dir, data_prc_cfg.filename_check_data_preproc_logging_txt))
    figsize = data_prc_cfg.figsize
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, data_prc_cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Create folder if not exist
        create_folder_if_not_exists(figures_dir)
        create_folder_if_not_exists(figures_all_dir)

        # Get patient_ids
        patients_list = os.listdir(os.path.join(data_prc_cfg.save_dir_dataset_full))
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # Testing for a small number of patients
        if data_prc_cfg.test_patients_list is not None:
            test_patients_list = data_prc_cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

      
        with Pool(processes = data_prc_cfg.workers) as pool:
            results = tqdm(pool.imap(multiproc_make_plots, zip(itertools.repeat(df_data_folder_types), patients_list)), total=len(patients_list))
            tuple(results)
        #for patient_id in tqdm(patients_list):
        #   make_plots(df_data_folder_types, patient_id)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')

def multiproc_make_plots(a_b_c):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return make_plots(*a_b_c)


def make_plots(df_data_folder_types, patient_id):

    figures_dir = data_prc_cfg.save_dir_figures
    figures_all_dir = data_prc_cfg.save_dir_figures_all
    figsize = data_prc_cfg.figsize

    save_dir_dataset_full = data_prc_cfg.save_dir_dataset_full
    #logger.my_print('Patient_id: {id}'.format(id=patient_id))

    # Determine folder type of interest ('with_contrast'/'no_contrast')
    # TODO: temporary, for MDACC
    if data_prc_cfg.use_umcg:
        folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
    else:
        folder_type_i = ''

    # Create folder if not exist
    create_folder_if_not_exists(os.path.join(figures_dir, patient_id))

    # Load npy data of shape (1, z, y, x)
    ct_arr = np.load(os.path.join(save_dir_dataset_full, patient_id, data_prc_cfg.filename_ct_npy))[0]
    rtdose_arr = np.load(os.path.join(save_dir_dataset_full, patient_id, data_prc_cfg.filename_rtdose_npy))[0]
    segmentation_map_arr = np.load(os.path.join(save_dir_dataset_full, patient_id,
                                                        data_prc_cfg.filename_segmentation_map_npy))[0]

    # CT (with Windowing)
    # ct_arr = arr[0]
    ct_min_plot = data_prc_cfg.ct_wl - data_prc_cfg.ct_ww / 2
    ct_max_plot = data_prc_cfg.ct_wl + data_prc_cfg.ct_ww / 2
    # Source: https://myctregistryreview.com/courses/my-ct-registry-review-demo/lessons/ct-physics/topic/window-width-and-window-level/
    ct_arr = np.clip(ct_arr, a_min=ct_min_plot, a_max=ct_max_plot)
    plot_arr(ct_arr, nr_images=data_prc_cfg.plot_nr_images, figsize=figsize, cmap=data_prc_cfg.ct_cmap,
                     colorbar_title=data_prc_cfg.ct_colorbar_title,
                     filename=os.path.join(figures_dir, patient_id, 'ct.png'), ticks_steps=data_prc_cfg.ct_ticks_steps,
                     segmentation_map=segmentation_map_arr)

    # RTDOSE
    # rtdose_arr = arr[1]
    plot_arr(rtdose_arr, nr_images=data_prc_cfg.plot_nr_images, figsize=figsize, cmap=data_prc_cfg.rtdose_cmap,
                     colorbar_title=data_prc_cfg.rtdose_colorbar_title,
                     filename=os.path.join(figures_dir, patient_id, 'rtdose.png'),
                     vmin=data_prc_cfg.rtdose_vmin, vmax=data_prc_cfg.rtdose_vmax,
                     ticks_steps=data_prc_cfg.rtdose_ticks_steps,
                     RTdose=True)  # NOTE: Daniel: this boolean is needed to plot the RTDOSE with the correct colormap

    # segmentation_map
    plot_arr(segmentation_map_arr > 0, nr_images=data_prc_cfg.plot_nr_images, figsize=figsize,
                     cmap=data_prc_cfg.segmentation_cmap, colorbar_title=data_prc_cfg.segmentation_colorbar_title,
                     filename=os.path.join(figures_dir, patient_id, 'segmentation.png'),
                     ticks_steps=data_prc_cfg.segmentation_ticks_steps,
                     segmentation_map=segmentation_map_arr)

    plotting_rows_dicts = [
        {
            "Label": "CT,\nDose,\nSegmentation",
            "CT": ct_arr,
            "RTDOSE": rtdose_arr,
            "RTSTRUCT": segmentation_map_arr
        },
        {
            "Label": "CT",
            "CT": ct_arr,
        },
        {
            "Label": "Dose",
            "RTDOSE": rtdose_arr,
        },
        {
            "Label": "Segmentation",
            "RTSTRUCT": segmentation_map_arr
        }
    ]

    slices = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    fig, axes = plot_slices(plotting_rows_dicts, slices)# , title=f"{patient_id} slices")

    figures_folder_img_dir = os.path.join(figures_dir, patient_id, 'all.png')
    fig.savefig(figures_folder_img_dir)
    figures_all_folder_img_dir = os.path.join(figures_all_dir, '{}.png'.format(patient_id))
    fig.savefig(figures_all_folder_img_dir)
    plt.close(fig)
    


def main():
    #main_segmentation_outside_ct()
    main_plot()

if __name__ == '__main__':
    main()



