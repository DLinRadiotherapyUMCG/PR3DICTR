import numpy as np



def convert_to_sagittal(arr):
    sagittal_view = np.transpose(arr, (2, 1, 0))           # make sagittal view the first axis
    sagittal_view_rotated = np.rot90(sagittal_view, 1, axes=(1, 2))  # rotate the view
    return sagittal_view_rotated


def convert_to_coronal(arr):
    coronal_view = np.transpose(arr, (1, 2, 0))             # make coronal view the first axis
    coronal_view_rotated = np.rot90(coronal_view, 1, axes=(1, 2))   # rotate the view
    return coronal_view_rotated


def rotate_arrs_in_plotting_row_dicts(row_dicts, image_keys, plotting_axis):
    plotting_axis = plotting_axis.lower()
    for row_dict in row_dicts:
        for key, value in row_dict.items():
            if key in image_keys:   # find all possible images (CT, dose, structs, attention, etc.)
                if plotting_axis == "coronal":
                    row_dict[key] = convert_to_coronal(value)
                elif plotting_axis == "sagittal":
                    row_dict[key] = convert_to_sagittal(value)
                else:
                    raise ValueError(
                        f"Unknown plotting axis: {plotting_axis}. Must be 'axial', 'coronal', or 'sagittal'"
                    )
    return row_dicts


