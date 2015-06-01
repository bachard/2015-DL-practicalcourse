import gzip
import cPickle
import numpy
from matplotlib import pyplot as plt
import bhtsne.bhtsne as bhtsne
import os.path
import sys
import time

"""
Python procedure that produces a figure similar to the Figure 5
in paper L.J.P. van der Maaten. Barnes-Hut-SNE. (http://arxiv.org/pdf/1301.3342v2.pdf)
It uses Barnes-Hut implementation provided at http://lvdmaaten.github.io/tsne
(http://lvdmaaten.github.io/tsne/code/bh_tsne.tar.gz)
You need to compile bhtsne and place it along with the python wrapper in the foler bhtsne

:param NUM_SAMPLES: number of samples to produce the figure with
:param d: parameter that controls the spacing between digits
"""

start = time.time()

if len(sys.argv) >= 2:
    NUM_SAMPLES = min(int(sys.argv[1]), 70000)
if len(sys.argv) == 3:
    d = min(0.5, int(sys.argv[2]))
else:
    print("[ Using default number of samples 2000 and d=3]")
    NUM_SAMPLES = 2000
    d = 3

datafilename = "bhtsne{}.pkl.gz".format(NUM_SAMPLES)

# we check if the data has already been computed
if os.path.isfile(datafilename):
    print("Loading data from {}...".format(datafilename))
    f = gzip.open(datafilename)
    X, Y = cPickle.load(f)
    f.close()

# if not we compute it    
else:
    print("Computing BH-tSNE on MNIST for {} samples...".format(NUM_SAMPLES))
    dataset = "../datasets/mnist.pkl.gz"
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    samples = train_set[0]
    
    if NUM_SAMPLES > 50000:
        samples = numpy.vstack((samples, valid_set[0]))

    
    if NUM_SAMPLES > 60000:
        samples = numpy.vstack((samples, test_set[0]))
    
    X_values = samples[0:NUM_SAMPLES]
    X = numpy.asarray(X_values, dtype='float64')
    Y_values = []
    
    for y in bhtsne.bh_tsne(X):
        Y_values.append(y)
    
    Y = numpy.asarray(Y_values)
    print("Writing data to {}...".format(datafilename))
    f = gzip.open(datafilename, 'wb')
    cPickle.dump([X, Y], f)
    f.close()

    
Y *= 28 * d
delta = 30 # min 14
min_x = min(Y[:,0])
min_y = min(Y[:,1])
max_x = max(Y[:,0])
max_y = max(Y[:,1])
print("Positions of extrema: ({}, {}) ({}, {})".format(min_x, min_y, max_x, max_y))
res = numpy.zeros((max_x - min_x + 2 * delta, max_y - min_y + 2 * delta))

def transform(x,y):
    """ Returns the upper left point from where to draw the sample
    :param x: x coordinate
    :param y: y coordinate
    """
    return (numpy.floor(x - min_x) - 14 + delta, numpy.floor(y - min_y) - 14 + delta)

for (x,y) in zip(X,Y):
    orig_x, orig_y = transform(y[0], y[1])
    x = x.reshape((28, 28))
    for i in range(28):
        for j in range(28):
            if x[i, j] != 0:
                res[orig_x + i, orig_y + j] = 1

print("Matrix image size: {}".format(res.shape))

plt.set_cmap('binary')
plt.figure(figsize=(res.shape[0]/300, res.shape[1]/300))
plt.axis("off")
plt.imshow(res)
plt.savefig('bhtsne{}.png'.format(NUM_SAMPLES), dpi=300, bbox_inches='tight')

end = time.time()
print("Total running time: {}s".format(end-start))
