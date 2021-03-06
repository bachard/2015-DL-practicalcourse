import theano
import theano.tensor as T
import numpy
import cPickle
import gzip
import PIL
import os.path
from scipy.misc import imresize
from matplotlib import pyplot as plt

import time
# We import the utils file
import sys
sys.path.append("..")
import utils

class KMeans(object):
    """ Theano implementation of k-Means following the paper by Adam Coates
    (http://ai.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf)
    """
    def __init__(self, input, k, e_zca):
        """
        :param input: input (theano.tensor.matrix)
        :param k: number of centroids
        :param e_zca: epsilon parameter used in ZCA whitening
        :attribute centroids: theano shared variable containing the centroids
               
        :function normalize_input: theano function computing normalized input (following previously cited paper)
        :function whiten_input: theano function computing ZCA whitening of the input (following previously cited paper)
        :function update_centroids: theano function computing the updates of the centroids (following previously cited paper)
        """
        
        self.n_samples, self.n_dims = input.shape
        self.k = k
        self.e_zca = e_zca
        self.normalize_input = self._normalize_input()
        self.whiten_input = self._whiten_input(self.n_samples)

        # in the process the input is rotated: the columns contains the samples
        self.x = numpy.asarray(self.whiten_input(self.normalize_input(input)), dtype=theano.config.floatX)

        # initialize centroids: normally distributed random vectors on the unit sphere
        # according to the pdf
        # first we draw normally distributed random vectors
        centroids_values = numpy.random.normal(0, 1, (self.n_dims, self.k))
        # then we normalize them
        centroids_values = centroids_values / numpy.sqrt((centroids_values ** 2).sum(0))

        self.centroids_values = numpy.asarray(centroids_values, dtype=theano.config.floatX)
        
        self.update_centroids = self._update_centroids()

        
    def _normalize_input(self):
        X = T.matrix('X')
        results, updates = theano.scan(
            lambda x_i: (x_i - T.mean(x_i)) / T.sqrt(T.var(x_i) + 10),
            sequences=[X]
        )
        return theano.function(inputs=[X], outputs=results)
        

    def _whiten_input(self, n): 
        X = T.matrix('X', dtype=theano.config.floatX)

        cov = T.dot(X.T, X) / (n - 1)
        
        eigenvalues, eigenvectors = T.nlinalg.eig(cov)

        V = eigenvectors
        D = eigenvalues
        D_prime = T.nlinalg.alloc_diag(T.inv(T.sqrt(D + self.e_zca)))
                
        M = T.dot(V, T.dot(D_prime, V.T))

        # now the input has been rotated: each column is a sample
        return theano.function(inputs=[X], outputs=T.dot(M, X.T))


    def _update_centroids(self):
        X = T.matrix('X')
        D = T.matrix('D')
        zeros = numpy.zeros((self.n_samples, self.k), dtype=theano.config.floatX)
        output = T.matrix("output")
        
        def set_value(i, j, value, prior_output):
            return T.set_subtensor(prior_output[i, j], value)

        prod = T.dot(D.T, X)
        rows = T.arange(self.n_samples)
        indices = T.argmax(T.abs_(prod), axis=0)

        result, updates = theano.scan(
            fn=set_value,
            outputs_info=output,
            sequences=[rows, indices, prod[indices, rows]],
        )

        S = result[-1].T

        D_new = T.dot(X, S.T) + D
        D_new = D_new / D_new.norm(L=2, axis=0)

        return theano.function(
            inputs=[X, D],
            outputs=D_new,
            givens={output: zeros}
        )
    

def to_rgb_array(img, size):
    """Correctly reshape an image containing RGB values represented as a vector,
    e.g. CIFAR-10 images, to be in the same shape as PIL/numpy convention
    useful for resize and show image
    """
    return numpy.asarray([(img[i], img[i + size], img[i + 2 * size]) for i in range(size)])
        
def from_rgb_array(img):
    """Reverse operation of previous function"""
    r = img[0::3]
    g = img[1::3]
    b = img[2::3]
    return numpy.concatenate([r, g, b])


def train_kmeans(k=500, e_zca=0.01, n_iters=10, n_cifar_batches=5):
    """
    Trains a k-Means model on CIFAR-10 dataset, resizing the image from 32x32 to 12x12
    
    :param k: number of centroids
    :param e_zca: epsilon parameter used in ZCA whitening
    :param n_iters: number of iterations to update centroids
    :param n_cifar_batches: number of CIFAR-10 batches to use
    """
    # First we gather and resize the images
    # for convenience we save the resized CIFAR images in a pkl.gz file
    data = None

    resized_size = 12
    
    if os.path.isfile("CIFAR_resized.pkl.gz"):
        print("Loading resized images from CIFAR_resized.pkl.gz...")
        f = gzip.open("CIFAR_resized.pkl.gz")
        resized_data = cPickle.load(f)
        f.close() 
    
    else:
        print("Resizing CIFAR-10 images...")
        for i in range(1, n_cifar_batches+1):
            f = open("../datasets/cifar-10-batches-py/data_batch_{}".format(i))
            dict = cPickle.load(f)
            f.close()
            if data is None:
                data = dict["data"]
            else:
                data = numpy.vstack((data, dict["data"]))
            
    
        resized_data = []
    
        for img in data:
            img = to_rgb_array(img, 32 * 32)
            img = img.reshape((32, 32, 3))
            img = imresize(img, (resized_size,resized_size))
            img = img.reshape(resized_size * resized_size * 3)
            img = from_rgb_array(img)
            resized_data.append(img.reshape(resized_size * resized_size * 3))
        print("Writing resized images to CIFAR_resized.pkl.gz...")
        f = gzip.open("CIFAR_resized.pkl.gz", "wb")
        cPickle.dump(resized_data, f)
        f.close()
    

    # Then we initialize the model
    x = numpy.asarray(resized_data, dtype=theano.config.floatX)
    n_samples, n_dims = x.shape

    print("Initializing model...")
    kmeans = KMeans(x, k, e_zca)

    print("Starting k-Means training...")
    for i in range(n_iters):
        print("Iteration {}...".format(i+1))
        kmeans.centroids_values = kmeans.update_centroids(
            kmeans.x,
            kmeans.centroids_values
        )
        
    # final centroids
    D = kmeans.centroids_values.T

    print("Saving receptive fields to repflds.png...")
    # same function as in utils.py modified for RGB images
    n_h, n_w = utils.layout_shape(k)

    receptive_fields = numpy.zeros((n_h * resized_size, n_w * resized_size, 3))

    #def rescale_rgb(D):

    def f(x, a, b):
        return (x-a) / (b-a)
    rescale = numpy.vectorize(f)

    D = rescale(D, D.min(), D.max())

    for i in range(n_h):
        for j in range(n_w):
            img = D[i * n_w + j]
            img = to_rgb_array(img, resized_size * resized_size)
            img = img.reshape((resized_size, resized_size, 3))
            receptive_fields[i * resized_size: (i + 1) * resized_size, j * resized_size: (j + 1) * resized_size, :] = img

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    xticks = numpy.arange(0, n_w * resized_size, resized_size)                                              
    yticks = numpy.arange(0, n_h * resized_size, resized_size)
    ax.set_xticks(xticks)
    ax.set_xticklabels([i for (i,x) in enumerate(xticks)]) 
    ax.set_yticks(yticks)
    ax.set_yticklabels([i for (i,y) in enumerate(yticks)])
    ax.grid(which="both", linestyle='-')
    plt.set_cmap("gray")
    plt.imshow(receptive_fields)
    plt.savefig("repflds.png", dpi=300)
    
if __name__ == "__main__":
    start = time.time()
    train_kmeans(
        k=500,
        e_zca=0.05,
        n_iters=10,
        n_cifar_batches=5
    )
    end = time.time()
    print("Total running time: {}s".format(end-start))
