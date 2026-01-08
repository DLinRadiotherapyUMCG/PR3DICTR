

from .ModalityPlotter import ModalityPlotter    


class DefaultPlotter(ModalityPlotter):
    """ Default plotting strategy (just a placeholder)"""
    def plot(self, axs, data, slices, params, **kwargs):

        for i, slice_n in enumerate(slices):
            axs[i].imshow(data[slice_n], cmap='viridis', interpolation='none')