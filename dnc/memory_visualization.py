import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# plt.switch_backend("TkAgg")
print("matplotlib backernd", matplotlib.get_backend())


def visualize_op(input_series, dnc_output, memory_view, dirname=None, step=0,
                 figsize=(16, 12), show=False, savefig=True):
    """Visualize memory state of DNC. It was constructed to work for copy task
    with one read head, so may not work as expected for other tasks
    Args:
        input_series: 2D or 3D numpy array. In case of 3D array 1st dimension
            should be equal to 1.
        dnc_output: 2D or 3D numpy array. Same as input_series.
        memory_view: python dict of memory view from DNC. It received by call
            in tensorflow `output, memory_view = DNC_instance.get_outputs()`
            and after fetched with session.
        dirname: str, where image should be saved
        step: int, step for image saving
        figsize: 2D tuple of int, size of the figure
        show: bool, show image or not
        savefig: bool, save image or not
    """
    fig = plt.figure(1, figsize=figsize)

    input_series = np.transpose(np.squeeze(input_series))
    axe = plt.subplot(611)
    plt.title("inputs")
    plt.imshow(input_series, cmap="gray", interpolation='nearest')
    axe.set_aspect('auto')

    dnc_output = np.transpose(np.squeeze(dnc_output))
    axe = plt.subplot(612)
    plt.title("output")
    plt.imshow(dnc_output, cmap="gray", interpolation='nearest')
    axe.set_aspect('auto')

    ww_strip = np.squeeze(memory_view['write_weightings'])
    rw_strip = np.squeeze(memory_view['read_weightings'])
    print(
        "\t\nmax write", np.max(ww_strip), "max read", np.max(rw_strip),
        "mean write", np.mean(ww_strip), "mean read", np.mean(rw_strip))
    colored_write = np.zeros((ww_strip.shape[0], ww_strip.shape[1], 3))
    colored_read = np.zeros((rw_strip.shape[0], rw_strip.shape[1], 3))
    for i in range(ww_strip.shape[0]):
        for j in range(ww_strip.shape[1]):
            colored_read[i, j] = [rw_strip[i,j], 0., 0.]
            colored_write[i, j] = [0., ww_strip[i,j], 0.]
    axe = plt.subplot(613)
    plt.imshow(np.transpose(colored_write + colored_read, [1, 0, 2]), interpolation='nearest')
    plt.title("Memory Location")
    write_legend = mpatches.Rectangle((1,1), 1, 1, color='green', label='Write Head')
    read_legend = mpatches.Rectangle((1,1), 1, 1, color='red', label='Read Head')
    plt.legend(bbox_to_anchor=(0.0, 0.5), handles=[write_legend, read_legend])
    axe.set_aspect('auto')

    free_strip = np.vstack([np.squeeze(memory_view['free_gates'])] * 5)
    axe = plt.subplot(614)
    axe.imshow(free_strip, cmap=plt.cm.gray, interpolation='nearest')
    axe.set_title("Free Gate")
    axe.set_yticks([])
    axe.set_aspect('auto')

    allocation_strip = np.vstack([np.squeeze(memory_view['allocation_gates'])] * 5)
    axe = plt.subplot(615)
    axe.imshow(allocation_strip, cmap=plt.cm.gray, interpolation='nearest')
    axe.set_title("Alloc. Gate")
    axe.set_yticks([])
    axe.set_aspect('auto')

    axe = plt.subplot(616)
    axe.imshow(np.squeeze(memory_view['usage_vectors'].T), cmap=plt.cm.gray, interpolation='nearest')
    axe.set_title("Memory Locations Usage")
    axe.set_xlabel("Time")
    axe.set_aspect('auto')

    fig.tight_layout()
    if savefig:
        plt.savefig(os.path.join(dirname, "%d_image.png" % step))
    if show:
        plt.show()
