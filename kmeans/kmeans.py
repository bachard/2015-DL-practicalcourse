import theano
import theano.tensor as T
import numpy
import cPickle
import gzip
import PIL
import os.path
from scipy.misc import imresize
from matplotlib import pyplot as plt
class KMeans(object):

    """


    """

    def __init__(self, input, k):

        """
        We transpose the input according
        """
        self.n_samples, self.n_dims = input.shape
        self.k = k

        self.normalize_input = self._normalize_input()
        self.whiten_input = self._whiten_input(self.n_samples)

        # in the process the input is rotated: the columns are the samples
        self.x = numpy.asarray(self.whiten_input(self.normalize_input(input)), dtype=theano.config.floatX)

        # initialize centroids: normally distributed random vectors on the unit sphere
        # according to the pdf
        # first we draw normally distributed random vectors
        centroids_values = numpy.random.normal(0, 1, (self.n_dims, self.k))
        # then we normalize them
        centroids_values = centroids_values / numpy.sqrt((centroids_values ** 2).sum(0))

        self.centroids = theano.shared(
            value=numpy.asarray(centroids_values, dtype=theano.config.floatX),
            borrow=True
        )
        
        self.S = theano.shared(
            numpy.zeros((self.n_samples, self.k), dtype=theano.config.floatX),
            borrow=True
        )

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

        e_zca = 0.01

        cov = T.dot(X.T, X) / (n - 1)
        
        eigenvalues, eigenvectors = T.nlinalg.eig(cov)

        V = eigenvectors
        D = eigenvalues
        D_prime = T.nlinalg.alloc_diag(T.inv(T.sqrt(D + e_zca)))
                
        M = T.dot(V, T.dot(D_prime, V.T))

        # now the input has been rotated: each column is a sample
        return theano.function(inputs=[X], outputs=T.dot(M, X.T))


    def _update_centroids(self):

        X = T.matrix('X')
        D = T.matrix('D')
        zeros = numpy.zeros((self.n_samples, self.k), dtype=theano.config.floatX)
        
        # D[T.arange(self.n_samples), T.argmax(T.abs_(T.dot(self.centroids.T, X)), axis=0).eval()]
        output = T.matrix("output")

        def set_value(i, j, value, prior_output):
            return T.set_subtensor(prior_output[i, j], value)

        prod = T.dot(D.T, X)
        rows = T.arange(self.n_samples)
        indices = T.argmax(T.abs_(prod), axis=0)

        """
        L'immondite n'est rien face a ces 4 lignes de code
        """
        result, updates = theano.scan(
            fn=set_value,
            outputs_info=output,
            sequences=[rows, indices, prod[indices, rows]],
        )

        S = result[-1].T

        D_new = T.dot(X, S.T) + D
        D_new = D_new / D_new.norm(L=2, axis=0)

        return theano.function(
            inputs=[],
            outputs=D_new,
            givens={X: self.x, D: self.centroids, output: zeros}
        )
    

def to_rgb_array(img, size):
    return numpy.asarray([(img[i], img[i + size], img[i + 2 * size]) for i in range(size)])
        
if __name__ == "__main__":

    data = None
    
    if os.path.isfile("CIFAR_resized.pkl.gz"):
        print("Loading resized images from CIFAR_resized.pkl.gz...")
        f = gzip.open("CIFAR_resized.pkl.gz")
        resized_data = cPickle.load(f)
        f.close() 
    
    else:
        print("Resizing CIFAR-10 images...")
        for i in range(1, 6):
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
            img = imresize(img, (12,12))
            resized_data.append(img.reshape(12 * 12 * 3))
        print("Writing resized images to CIFAR_resized.pkl.gz...")
        f = gzip.open("CIFAR_resized.pkl.gz", "wb")
        cPickle.dump(resized_data, f)
        f.close()
    
        
    x = numpy.asarray(resized_data, dtype=theano.config.floatX)
    n_samples, n_dims = x.shape
    kmeans = KMeans(x, 500)
        
    print("Init centroids:")
    print(kmeans.centroids.get_value())
    for i in range(10):
        print("Iteration {}".format(i+1))
        kmeans.centroids.set_value(kmeans.update_centroids())
    print("Final centroids:")
    print(kmeans.centroids.get_value())
    
    D = kmeans.centroids.get_value().T
    
    f = gzip.open("temp.pkl.gz", "wb")
    cPickle.dump(D, f)
    f.close()

    # f = gzip.open("temp.pkl.gz")
    # D = cPickle.load(f)
    # f.close()
    

    n_h = 20
    n_w = 25

    receptive_fields = numpy.zeros((n_h * 12, n_w * 12, 3))
    
    for i in range(n_h):
        for j in range(n_w):
            img = D[i * 12 + j]
            img = to_rgb_array(img, 12 * 12)
            img = img.reshape((12, 12, 3))
            receptive_fields[i * 12: (i + 1) * 12, j * 12: (j + 1) * 12, :] = img

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    xticks = numpy.arange(0, n_w * 12, 12)                                              
    yticks = numpy.arange(0, n_h * 12, 12)
    ax.set_xticks(xticks)                                                       
    ax.set_yticks(yticks)                                        
    ax.grid(which="both", linestyle='-')

    
    plt.imshow(receptive_fields)
    plt.savefig("repflds.png", dpi=300)
    
