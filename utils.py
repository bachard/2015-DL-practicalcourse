import theano
import numpy
from matplotlib import pyplot as plt


def visualize_matrix(matrix, n_imgs, imgsize, outputfile, cmap="gray", dpi=150):
    """

    """
    n_h,n_w = layout_shape(n_imgs)
    
    receptive_fields = numpy.zeros((n_h * imgsize, n_w * imgsize), dtype=theano.config.floatX)
    
    for i in range(n_h):
        for j in range(n_w):
            img = matrix[i * n_w + j]
            img = img.reshape((imgsize, imgsize))
            receptive_fields[i * imgsize: (i + 1) * imgsize, j * imgsize: (j + 1) * imgsize] = img
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1,1,1)
    xticks = numpy.arange(0, n_w * imgsize, imgsize)                                              
    yticks = numpy.arange(0, n_h * imgsize, imgsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels([i for (i,x) in enumerate(xticks)])
    ax.set_yticks(yticks)
    ax.set_yticklabels([i for (i,y) in enumerate(yticks)])
    ax.grid(which="both", linestyle='-')
    
    plt.set_cmap(cmap)
    plt.imshow(receptive_fields)
    plt.savefig(outputfile, dpi=dpi)


def layout_shape(n):
    """Returns a layout shape given a number of images to display with a ratio n_horizontal / n_vertical the closest to 1
    e.g. 100 -> (10,10); 300 -> (15, 20); etc.    
    """
    return [(i, n / i) for i in range(1, int(n**0.5)+1) if n % i == 0][-1]
