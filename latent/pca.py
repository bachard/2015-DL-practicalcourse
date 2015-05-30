import numpy
import theano
import theano.tensor as T
import gzip
import cPickle
from matplotlib import pyplot as plt
import sys

class PCA(object):

    """PCA class implementation using SVD
    
    """
        
    def __init__(self, n_dims):
        """Initialize the parameters of PCA

        :param n_dims: number of components to keep
        
        :function compute_pca(input): theano function computing 
        the n_dims principal components of input using SVD 
        (theano.tensor.nlinalg.svd runs on CPU)
        """
        self.n_dims = n_dims
        self.compute_pca = self._compute_pca()
        
        
    def _compute_pca(self):
        X = T.matrix('X')
        n_samples, n_in = X.shape
        mean = T.mean(X, axis=0)
        # before computing SVD we center the input
        X_centered = X - mean

        U, S, V = T.nlinalg.svd(X_centered, full_matrices=0)
        V = V.T
        V_restricted = V[:, 0:self.n_dims]
        X_pca = T.dot(X_centered, V_restricted)
                
        return theano.function(
            inputs=[X],
            outputs=X_pca
        )

def plot_mnist(dataset="../datasets/mnist.pkl.gz", outputfile="scatterplotMNIST.png"):
    """Compute scatterplot for MNIST dataset"""
    
    print("Gathering MNIST data...")
    f = gzip.open(dataset, "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    x_train, y_train = train_set

    print("Initializing PCA...")
    pca = PCA(2)
    print("Constructing scatterplot...")
    scatterplot(pca, x_train, y_train, 10, outputfile)
    

def plot_cifar(dataset="../datasets/cifar-10-batches-py", outputfile="scatterplotCIFAR.png"):
    """Compute scatterplot for CIFAR-10 dataset"""
    
    print("Gathering CIFAR data...")
    x_train = None
    y_train = None
    for i in range(1, 6):
        f = open("../datasets/cifar-10-batches-py/data_batch_{}".format(i))
        dict = cPickle.load(f)
        f.close()
        if x_train is None:
            x_train = dict["data"]
            y_train = dict["labels"]
        else:
            x_train = numpy.vstack((x_train, dict["data"]))
            y_train.append(dict["labels"])

    print("Initializing PCA...")
    pca = PCA(2)
    print("Constructing scatterplot...")
    scatterplot(pca, x_train, y_train, 10, outputfile)


def scatterplot(pca, x_train, y_train, n_classes, outputfile):
    """Construct scatterplot of (x_train, y_train) data
    
    :param pca: PCA object used for computing PCA on input data
    :param x_train: input data
    :param y_train: input labels
    :param n_classes: number of classes of input data
    """
    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i > j:
                continue
            print("Computing PCA for pair {}".format((i,j)))
            x = numpy.asarray([x for (x, y) in zip(x_train, y_train) if y==i or y==j])           
            y = ([y for y in y_train if y==i or y==j])
            x_pca = pca.compute_pca(x)
            
            plots[i, j].scatter(x_pca[:, 0], x_pca[:, 1], c=y)
            plots[i, j].set_xticks(())
            plots[i, j].set_yticks(())

            plots[j, i].scatter(x_pca[:, 0], x_pca[:, 1], c=y)
            plots[j, i].set_xticks(())
            plots[j, i].set_yticks(())

            if i == 0:
                plots[i, j].set_title(j)
                plots[j, i].set_ylabel(j)
                
    plt.tight_layout()
    print("Saving figure...")
    plt.savefig(outputfile)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Wrong number of parameters (usage: pca.py dataset_name (dataset_name=MNIST or CIFAR))")
    else:
        if sys.argv[1] == "MNIST":
            plot_mnist()
        elif sys.argv[1] == "CIFAR":
            plot_cifar()
        else:
            print("This dataset is unknown (available datasets: MNIST, CIFAR)")
