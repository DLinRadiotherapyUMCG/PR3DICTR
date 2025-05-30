import napari
import numpy as np
import os
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
from napari.utils.colormaps import Colormap
import matplotlib.pyplot as plt
import src.constants as constants
from src.visualization.rtdose_colormap import create_RTDOSE_cmap

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from qtpy.QtGui import QColor
from qtpy.QtCore import Qt



"""
TODO:
- add a legend for the segmetation maps and CT
- make the legend change depending on the chosen layer
- change the segmentations to just be an outline?
- be able to select which segmentations to display
- improve dropdown menu to select patients (possibly with a search bar)
- dropdown to select the axis to display (axial, coronal, sagittal)
- how to set the data directory
- change opacity of dose if the CT is not displayed
"""

DATA_FOLDER = r"H:\MDACC_higherres\dataset_full_mdacc"

class DiscreteColorbar(QWidget):
    def __init__(self, levels, colors, labels=None, title="Colorbar"):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignTop)

        if title:
            title_label = QLabel(f"<b>{title}</b>")
            self.layout().addWidget(title_label)

        if labels is None:
            labels = [f"{levels[i]}–{levels[i+1]}" for i in range(len(levels) - 1)]

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

class NapariRTViewer:
    def __init__(self, data_folder=DATA_FOLDER):
        self.data_folder = data_folder
        self.viewer = None

    def load_patient_data(self, patient_id):
        patient_folder = os.path.join(self.data_folder, patient_id)
        ct = np.load(os.path.join(patient_folder, "ct.npy"))[0]
        dose = np.load(os.path.join(patient_folder, "rtdose.npy"))[0]
        seg = np.load(os.path.join(patient_folder, "segmentation_map.npy"))[0]

        #ct = np.clip(ct, -200, 400)

        return ct, dose, seg

    def list_patients(self, _=None):
        return sorted([d for d in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, d))])

    def make_napari_colormap_from_matplotlib(self, cmap, norm, levels, name='discrete_cmap'):
        colors = [cmap(norm(level)) for level in levels]
        colors[0] = (0, 0, 0, 0)
        colors = colors[:-1]
        value_bins = levels
        vmin, vmax = value_bins[0], value_bins[-1]
        normalized_bins = [(b - vmin) / (vmax - vmin) for b in value_bins]

        step_colors = []
        step_controls = []
        for i in range(len(colors)):
            step_colors.extend([colors[i], colors[i]])
            step_controls.extend([normalized_bins[i], normalized_bins[i+1]])
        
        napari_cmap = Colormap(colors=step_colors, controls=step_controls, name=name)
        return napari_cmap

    def launch(self):
        self.viewer = napari.Viewer(title="RTViewer")
        self.viewer.dims.ndisplay = 2
        self.viewer.dims.order = (0, 1, 2)

        @magicgui(
            auto_call=False,
            patient_id={"choices": self.list_patients},
            layout='horizontal',
            call_button="Load Patient"
        )
        def load_and_display(patient_id: str):
            try:
                ct, dose, seg = self.load_patient_data(patient_id)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to load patient data: {e}")
                return

            self.viewer.layers.clear()
            patient_type = "HNC"
            params = constants.PLOTTING_PARAMS[patient_type]

            ct_cmap = params["CT"]["cmap"]

            dose_cmap, dose_norm, dose_levels = create_RTDOSE_cmap(patient_type)
            dose_levels = [level for level in dose_levels if level <= 8000]
            if dose_levels[-1] < 8000:
                dose_levels.append(8000)
            normalised_dose_cmap = self.make_napari_colormap_from_matplotlib(
                cmap=dose_cmap, norm=dose_norm, levels=dose_levels, name='RTDOSE_cmap'
            )
            self.viewer.add_image(
                ct,
                name='CT',
                colormap=ct_cmap,
                blending='opaque',
                cache=True,
                contrast_limits=[-200, 400],
                scale=(2,1,1)
            )
            self.viewer.add_image(
                dose,
                name='RTDOSE',
                colormap=normalised_dose_cmap,
                blending='translucent',
                contrast_limits=[0, 8000],
                opacity=0.3,
                cache=True,
                scale=(2,1,1)
            )
            self.viewer.add_labels(seg, name='SEGMENTATIONS', opacity=0.5, scale=(2,1,1))
            dose_levels_widget = dose_levels[1:-1]
            dose_colors = [dose_cmap(dose_norm(level)) for level in dose_levels_widget]
            
            widget = DiscreteColorbar(
                levels=dose_levels_widget,
                colors=dose_colors,
                labels=[str(x) for x in dose_levels_widget],
                title="Dose (Gy)"
            )
            self.viewer.window.add_dock_widget(widget, area="right")

        self.viewer.window.add_dock_widget(load_and_display, area="left")
        napari.run()

if __name__ == "__main__":
    NapariRTViewer().launch()
