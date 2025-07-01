import napari
import numpy as np
import os
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
from napari.utils.colormaps import Colormap
import matplotlib.pyplot as plt
import src.constants as constants
from src.visualization.rtdose_colormap import create_RTDOSE_cmap


DATA_FOLDER = r"H:\MDACC_higherres\dataset_full_mdacc"


from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from qtpy.QtGui import QColor
from qtpy.QtCore import Qt


class DiscreteColorbar(QWidget):
    def __init__(self, levels, colors, labels=None, title="Colorbar"):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignTop)

        if title:
            title_label = QLabel(f"<b>{title}</b>")
            self.layout().addWidget(title_label)

        if labels is None:
            # Default labels from level intervals
            labels = [f"{levels[i]}–{levels[i+1]}" for i in range(len(levels) - 1)]

        # Reverse the order of swatches and labels
        for i, label in reversed(list(enumerate(labels))):
            swatch = self._make_color_swatch(colors[i], label)
            self.layout().addWidget(swatch)

    def _make_color_swatch(self, rgb, label):
        frame = QFrame()
        frame.setFixedHeight(20)
        frame.setStyleSheet(
            f"background-color: rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)});"
            f"border: 1px solid black;"
        )
        text = QLabel(label)
        text.setStyleSheet("padding-left: 4px;")
        container = QWidget()
        container.setLayout(QVBoxLayout())
        container.layout().setContentsMargins(0, 0, 0, 0)
        container.layout().addWidget(frame)
        container.layout().addWidget(text)
        return container





def load_patient_data(patient_id):
    patient_folder = os.path.join(DATA_FOLDER, patient_id)
    ct = np.load(os.path.join(patient_folder, "ct.npy"))[0]
    dose = np.load(os.path.join(patient_folder, "rtdose.npy"))[0]
    seg = np.load(os.path.join(patient_folder, "segmentation_map.npy"))[0]

    print(dose.min(), dose.max())
    
    ct = np.clip(ct, -200, 400)
    #dose = np.clip(dose, 0, 8000)  # Assuming dose is in the range [0, 100] for visualization

    print(dose.min(), dose.max())

    return ct, dose, seg

def list_patients(_=None):
    return sorted([d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))])


def make_napari_colormap_from_matplotlib(cmap, norm, levels, name='discrete_cmap'):

    colors = [cmap(norm(level)) for level in levels]
    colors[0] = (0, 0, 0, 0)  # Set the first color to transparent
    colors = colors[:-1]  # Exclude the last color to avoid an extra step

    value_bins = levels  # Use the levels as value bins
    vmin, vmax = value_bins[0], value_bins[-1]
    normalized_bins = [(b - vmin) / (vmax - vmin) for b in value_bins]

    step_colors = []
    step_controls = []

    print(colors)
    print(value_bins)

    for i in range(len(colors)):
        step_colors.extend([colors[i], colors[i]])  # repeat color
        step_controls.extend([normalized_bins[i], normalized_bins[i+1]])  # start and end of bin

    print(step_colors)
    print(step_controls)

    # Create the napari Colormap
    napari_cmap = Colormap(colors=step_colors, controls=step_controls, name=name)

    return napari_cmap

def launch_napari_with_dropdown():
    viewer = napari.Viewer(title="RTViewer")
    viewer.dims.ndisplay = 2
    viewer.dims.order = (0, 1, 2)
    # By default, napari allows scrolling through slices with the mouse wheel when hovering over the image.
    # To ensure this, we do NOT set any mouse_scroll_action or require ctrl.
    # If you previously changed napari settings, reset them to default for normal scroll behavior.
    # No further action is needed here for standard napari behavior.

    @magicgui(
        auto_call=True,
        patient_id={"choices": list_patients},
        layout='horizontal',
        call_button="Load Patient"
    )
    def load_and_display(patient_id: str):
        try:
            ct, dose, seg = load_patient_data(patient_id)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load patient data: {e}")
            return

        viewer.layers.clear()
        # Use colormaps from plot_slices.py and constants.py
        # Choose patient type (HNC or LUNG) based on patient_id or add a selector if needed
        patient_type = "HNC"  # or "LUNG"; you can make this dynamic if needed
        params = constants.PLOTTING_PARAMS[patient_type]
        # CT colormap
        ct_cmap = plt.get_cmap(params["CT"]["cmap"])
        # Dose colormap (custom)
        dose_cmap, dose_norm, dose_levels = create_RTDOSE_cmap(patient_type)

        # Clip dose_levels to the range [0, 8000]
        dose_levels = [level for level in dose_levels if level <= 8000]
        if dose_levels[-1] < 8000:
            dose_levels.append(8000)  # Ensure 8000 is included as the last bin

        normalised_dose_cmap = make_napari_colormap_from_matplotlib(cmap=dose_cmap, norm=dose_norm, levels=dose_levels, name='RTDOSE_cmap')        
        # Add CT
        viewer.add_image(
            ct,
            name='CT',
            colormap='gray',
            blending='opaque',
            cache=True,
            scale=(2, 1, 1)
        )
        # Add Dose
        dose_layer = viewer.add_image(
            dose,
            name='RTDOSE',
            colormap=normalised_dose_cmap,
            blending='translucent',
            contrast_limits=[0, 8000],
            opacity=0.3,
            cache=True,
            scale=(2,1,1)
        )
        
        # Add Segmentations (use default or custom colormap if needed)
        #viewer.add_labels(seg, name='SEGMENTATIONS', opacity=0.5, scale=(2,1,1))

        # Create the widget
        dose_levels = dose_levels[1:-1]
        dose_colors = [dose_cmap(dose_norm(level)) for level in dose_levels]

        print("DOSE LABELS")
        print(dose_levels)
        widget = DiscreteColorbar(levels=dose_levels, colors=dose_colors, 
                                  labels=[str(x) for x in dose_levels], title="Dose (Gy)")
        viewer.window.add_dock_widget(widget, area="right")

        #viewer.window.remove_dock_widget(widget)

    viewer.window.add_dock_widget(load_and_display, area="left")
    napari.run()

    def make_colorbar_legend(self):
        pass

if __name__ == "__main__":
    launch_napari_with_dropdown()
