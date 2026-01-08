
from .ModalityPlotter import ModalityPlotter


class EmptyPlotter(ModalityPlotter):
    """ Plotting strategy for empty rows (i.e. no data to plot) """
    def set_colourmaps(self):
        self.cmap = None
        self.norm = None

    def plot(self, axs, _, _params, color="black", **kwargs):
        for ax in axs:
            ax.set_facecolor(color)