import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt

# Adjust this path to point to your root data folder
DATA_FOLDER = r"H:\preprocess_MT_data_UID_FULL\dataset_full"
DATA_FOLDER = r"H:\MDACC_higherres\dataset_full_mdacc"

def load_patient_data(patient_id):
    # NOTE: already removes the first dimension
    patient_folder = os.path.join(DATA_FOLDER, patient_id)
    ct = np.load(os.path.join(patient_folder, "ct.npy"))[0]
    dose = np.load(os.path.join(patient_folder, "rtdose.npy"))[0]
    seg = np.load(os.path.join(patient_folder, "segmentation_map.npy"))[0]
    return ct, dose, seg

from src.visualization.plot_slices import plot_slices

def plot_patient_slices_optional(patient_type, ct, dose, seg, slice_index, show_ct, show_dose, show_seg):

    plot_CT_dict = dict()

    
    
    if show_ct:
        plot_CT_dict['CT'] = ct
    if show_dose:
        plot_CT_dict['RTDOSE'] = dose
    if show_seg:
        plot_CT_dict['RTSTRUCT'] = seg

    slices = [slice_index]

    if plot_CT_dict:
        plot_CT_dict['Label'] = None
        fig, axes = plot_slices(row_dicts=[plot_CT_dict], slice_indexes=slices, title=None, RT_region=patient_type)
        st.pyplot(fig)
    else:
        st.info("Select at least one image to display.")

        



# Patch main to use the new plotting function and add checkboxes
def main():
    st.set_page_config(
        page_title="RTviewer",
        page_icon=":face_with_head_bandage:",
        #layout="wide"
        )

    st.title("3D Medical Image Viewer")
    
    patients = sorted([d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))])
    
    patient_id = st.selectbox("Select Patient", patients, index=0)

    patient_type = st.radio(
        "Select Patient Type",
        options=["HNC", "Lung"],
        index=0,
        horizontal=True
    )
    patient_type = patient_type.upper()  # Ensure patient type is uppercase
    
    if patient_id:
        ct, dose, seg = load_patient_data(patient_id)
        z_dim = ct.shape[0]
        slice_index = st.number_input(
        "Select Slice Index", 
        min_value=0, 
        max_value=z_dim - 1, 
        value=z_dim // 2, 
        step=1,
        format="%d"
    )
        col1, col2, col3 = st.columns(3)

        with col1:
            show_ct = st.checkbox("Show CT", value=True)
        with col2:
            show_dose = st.checkbox("Show Dose", value=True)
        with col3:
            show_seg = st.checkbox("Show Segmentations", value=True)

        plot_patient_slices_optional(patient_type, ct, dose, seg, slice_index, show_ct, show_dose, show_seg)



# if patient_id:
#     ct, dose, seg = load_patient_data(patient_id)
#     z_dim = ct.shape[0]
#     slice_index = st.number_input(
#         "Select Slice Index", 
#         min_value=0, 
#         max_value=z_dim - 1, 
#         value=z_dim // 2, 
#         step=1,
#         format="%d"
#     )
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         show_ct = st.checkbox("Show CT", value=True)
#     with col2:
#         show_dose = st.checkbox("Show Dose", value=True)
#     with col3:
#         show_seg = st.checkbox("Show Segmentations", value=True)

#     plot_patient_slices_optional(patient_type, ct, dose, seg, int(slice_index), show_ct, show_dose, show_seg)


if __name__ == "__main__":
    
    main()
    

    