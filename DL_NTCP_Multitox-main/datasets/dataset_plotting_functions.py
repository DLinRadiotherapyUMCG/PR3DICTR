import matplotlib as mpl
from matplotlib import colors
import math
import matplotlib.pyplot as plt
import numpy as np




def plotter(dir, patient_number):
    pass


def RT_plotter(stack):
    pass





"""
Subplots that show a selection of slices from one patient
"""

# sets the visual params for the RT slices
def define_RT_visuals(stack):
    upper_limit = 8600    # max value
    cmap = plt.cm.nipy_spectral

    bounds = np.linspace(0, upper_limit, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

# sets the visual params for the CT slices
def define_CT_visuals(stack):
    lower_limit = np.min(stack)
    upper_limit = np.max(stack)

    cmap = plt.cm.gray
    bounds = np.linspace(lower_limit, upper_limit, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

# sets the visual params for the SEGMAP slices
def define_SEGMAP_visuals(stack):
    lower_limit = np.min(stack)
    upper_limit = np.max(stack)

    cmap = plt.cm.hsv
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (0, 0, 0, 1.0) # force the first color entry to be black
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'SEGMAP cmap', cmaplist, cmap.N)
    
    bounds = np.linspace(lower_limit, upper_limit, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm



def show_slices(stack, RT=False, SEG=False, title = None, padding=0, rows=4, cols=4, save_dir=None):
    start_with = 0 + padding
    show_every = math.floor((stack.shape[0] - start_with - padding)/(rows*cols))              

    if RT: cmap, norm = define_RT_visuals(stack)
    elif SEG: cmap, norm = define_SEGMAP_visuals(stack)
    else: cmap, norm = define_CT_visuals(stack)
    
    # define the bins and normalize
    
    fig,ax = plt.subplots(rows,cols,figsize=[3*rows,3*cols])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows), int(i % rows)].imshow(stack[ind],cmap=cmap, norm=norm, interpolation='none')
        ax[int(i/rows), int(i % rows)].axis('off')
    fig.suptitle(title + f" ({stack.shape[0]} Slices)", fontsize = 13)
    plt.tight_layout()

    if save_dir != None:
        print(f"Saving Image! {title}")
        plt.savefig(save_dir + title + ".png")

    plt.show()









def show_all_three_inputs(CT, SEGMAP, RTDOSE, title=None, save_dir = None, rows=5, padding=0):
    cols = 3
    start_with = 0 + padding
    slice_spacing = math.floor((CT.shape[0] - start_with - padding) / rows)

    fig,ax = plt.subplots(rows,3,figsize=[2.3*cols, 2.5*rows])  # figsize is [width, height]
    

    cmap_CT, norm_CT = define_CT_visuals(CT)
    cmap_SEG, norm_SEG = define_SEGMAP_visuals(SEGMAP)
    cmap_RT, norm_RT = define_RT_visuals(RTDOSE)

    for i in range(rows):
        slice_idx = start_with + i*slice_spacing
        #ax[i, 0].set_title('slice %d' % ind)
        ax[i, 0].imshow(CT[slice_idx],cmap=cmap_CT, norm=norm_CT, interpolation='none')
        #ax[i, 0].axis('off')

        ax[i, 1].imshow(SEGMAP[slice_idx],cmap=cmap_SEG, norm=norm_SEG, interpolation='none')
        ax[i, 1].axis('off')

        ax[i, 2].imshow(RTDOSE[slice_idx],cmap=cmap_RT, norm=norm_RT, interpolation='none')
        ax[i, 2].axis('off')

        ax[i, 0].set(ylabel=f"Slice #{slice_idx}",xlabel=None)
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])


    if title != None: fig.suptitle(title + f" ({CT.shape[0]} Slices)", fontsize = 13)

    plt.tight_layout()

    if save_dir != None:
        print(f"Saving Image! {save_dir + title}.png")
        plt.savefig(save_dir + title + ".png")
    
    # Show the plot
    plt.show()

