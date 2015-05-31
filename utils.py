import theano
import numpy
from matplotlib import pyplot as plt


def visualize_matrix(matrix, n_h, n_w, imgsize, outputfile, cmap="gray", dpi=150):
    
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
