import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm, ListedColormap

from .ModalityPlotter import ModalityPlotter    


class ImagePlotter(ModalityPlotter):
    """ Plotting strategy for images (CT, PET, MR, etc.)"""

    def set_colourmaps(self, N = 256):
        cmap_name = self.modality_config["cmap"]
        min_val = self.modality_config["a_min"]
        max_val = self.modality_config["a_max"]

        self.cmap = mpl.colourmaps[cmap_name].resampled(N)
        self.norm  = Normalize(vmin=min_val, vmax=max_val)
        self.levels = None

    def plot(self, axs, image_array, slices, params, **kwargs):
        # plot the CT image slices along this axis of subplots
        for i, slice_n in enumerate(slices):
            axs[i].imshow(image_array[slice_n], cmap=self.cmap, norm=self.norm, interpolation='none')

