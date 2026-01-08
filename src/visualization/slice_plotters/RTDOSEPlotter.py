

from .rtdose_colormap import create_RTDOSE_cmap
from .ModalityPlotter import ModalityPlotter    

class RTDOSEPlotter(ModalityPlotter):
    """ Plotting strategy for RTDOSE images """

    def set_colourmaps(self, N = 256):
        #RT_region = self.RT_region
        self.cmap, self.norm, self.levels = create_RTDOSE_cmap(self.RT_region)
        self.levels = self.levels[1:]


    def plot(self, axs, RTDOSE, slices, params, is_background=False, **kwargs):
        # plot the RTDOSE image slices along this axis of subplots
        cmap = self.cmap
        norm = self.norm
        levels = self.levels


        for i, slice_n in enumerate(slices):
            slice_data = RTDOSE[slice_n]
            if is_background:
                # plot the RTDOSE as a background (i.e. the colours should not be transparent)
                axs[i].imshow(slice_data, cmap=cmap, norm=norm) # , interpolation='none')
                # axs[i].contourf(slice_data, cmap=cmap, norm=norm, levels=levels, alpha=1, origin="upper")
                axs[i].contour(slice_data, levels=levels, norm=norm, colors="white", linewidths=1, alpha=1) # , origin="upper") # contours are white
            else:
                # plot the RTDOSE with transparency (e.g. for on top of the CT scan)
                axs[i].contourf(slice_data, cmap=cmap, norm=norm, levels=levels, alpha=0.3) # , origin="lower")
                axs[i].contour(slice_data, cmap=cmap, norm=norm, levels=levels, linewidths=1) # , origin="lower") # contours are coloured

