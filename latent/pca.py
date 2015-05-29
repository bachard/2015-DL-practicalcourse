import numpy
import theano
import theano.tensor as T
import gzip
import cPickle
from random import randint
from matplotlib import pyplot as plt


from itertools import product
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

class PCAa(object):

    """
    
    steps:
    1) compute mean vector of inputs
    2) compute co-variance matrix
    3) compute eigenvectors and eigenvalues


    """
    
    def __init__(self):
        pass


def load_data(dataset):

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
        
def pca(dataset='mnist.pkl.gz'):

    X = T.matrix('X')
    
    datasets = load_data(dataset)

    x, y = datasets[0]

    n = x.shape[0].eval()
    d = x.shape[1].eval()
    
    # results, updates = theano.scan(lambda x_j: x_j.mean(), sequences=[X.T])
    # compute_mean_columns = theano.function(inputs=[], outputs=[results], givens={X: x})
    # 
    # mean = theano.shared(value=numpy.asarray(compute_mean_columns(), dtype=theano.config.floatX),
    #                      name='mean',
    #                      borrow=True)
    # 
    # results, updates = theano.scan(lambda x_j: x_j - mean, sequences=[X])
    # 
    # remove_mean = theano.function(inputs=[], outputs=[results], givens={X: x, mean: mean})
    # 
    # x = theano.shared(value=numpy.asarray(remove_mean()[0].reshape(n,d), dtype=theano.config.floatX),
    #                   name='x',
    #                   borrow=True)

    compute_xtx = theano.function(inputs=[], outputs=[T.dot(x.T, x)], givens={X: x})

    xtx = theano.shared(value=numpy.asarray(compute_xtx()[0], dtype=theano.config.floatX),
                        name='xtx',
                        borrow=True)
    
    compute_svd = theano.function(inputs=[], outputs=T.nlinalg.svd(X, full_matrices=0), givens={X: xtx})
    U, D, V = compute_svd()
    eigenvectors = U.T
    eigenvalues = D
    
    print(U.shape)
    print(V.shape)
    print(D.shape)

    print(U[0])
    
    eigenpairs = [(eigenvalues[i], eigenvectors[i]) for i in range(eigenvalues.size)]
    
    eigenpairs = sorted(eigenpairs, key=lambda x: x[0], reverse=True)
    
    # i = randint(0, d-1)
    # j = randint(0, d-1)
    # while j==i:
    #     j = randint(0, d-1)
    # 
    # print("Random values: {},{}".format(i, j))
    
    i,j = 0,1
    
    print(eigenpairs[i])
    print(eigenpairs[j])
    
    x, y = datasets[0]
    # pca = PCA(n_components=2)
    # x_pca =  pca.fit_transform(x.get_value())
    
    
    
    w_value = numpy.hstack((eigenpairs[i][1].reshape(d, 1), eigenpairs[j][1].reshape(d, 1)))
    
    x, y = datasets[0]
    
    W = w_value
    x = x.get_value()
    y = y.eval()
    x_pca = numpy.dot(x, W)
    classes = []
    for i in range(10):
        classes.append([(coor_x, coor_y) for ((coor_x, coor_y), c) in zip(x_pca, y) if c==i])
    
    for i in range(10):
        plt.clf()
        plt.scatter([x[0] for x in classes[i]], [x[1] for x in classes[i]], color='red')     
        plt.savefig('class_'+str(i)+'.png')
        
    # for i in range(10):
    #     for j in range(10):
    #         plt.clf()
    #         plt.scatter([x[0] for x in classes[i]], [x[1] for x in classes[i]], color='blue')
    #         plt.scatter([x[0] for x in classes[j]], [x[1] for x in classes[j]], color='red')     
    #         plt.savefig('class_'+str(i)+'_'+str(j)+'.png')


def pca_(input):

    X = T.matrix('X')
    
    x = theano.shared(value=input, name='x', borrow=True)

    n = x.shape[0].eval()
    d = x.shape[1].eval()
    
    results, updates = theano.scan(lambda x_j: x_j.mean(), sequences=[X.T])
    compute_mean_columns = theano.function(inputs=[], outputs=[results], givens={X: x})

    mean = theano.shared(value=numpy.asarray(compute_mean_columns(), dtype=theano.config.floatX),
                         name='mean',
                         borrow=True)

    results, updates = theano.scan(lambda x_j: x_j - mean, sequences=[X])

    remove_mean = theano.function(inputs=[], outputs=[results], givens={X: x, mean: mean})

    x = theano.shared(value=numpy.asarray(remove_mean()[0].reshape(n,d), dtype=theano.config.floatX),
                      name='x',
                      borrow=True)

    compute_xtx = theano.function(inputs=[], outputs=[T.dot(x.T, x)], givens={X: x})

    xtx = theano.shared(value=numpy.asarray(compute_xtx()[0], dtype=theano.config.floatX),
                        name='xtx',
                        borrow=True)
    
    compute_svd = theano.function(inputs=[], outputs=T.nlinalg.svd(X, full_matrices=0), givens={X: xtx})
    U, D, V = compute_svd()
    
    eigenvectors = U.T
    eigenvalues = D

    
    
    eigenpairs = [(numpy.abs(eigenvalues[i]), eigenvectors[i]) for i in range(eigenvalues.size)]

    eigenpairs = sorted(eigenpairs, key=lambda x: x[0], reverse=True)

    w_value = numpy.hstack((eigenpairs[0][1].reshape(d, 1), eigenpairs[1][1].reshape(d, 1)))

    x = input

    W = w_value
    
    x_pca = numpy.dot(x, W)

    return x_pca


def plot_class(dataset='mnist.pkl.gz'):

    datasets = load_data(dataset)

    x_train, y_train = datasets[0]

    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    
    for i in range(10):
        for j in range(10):
            print(i,j)
            x_ = numpy.asarray([x for (x, y) in zip(x_train.get_value(), y_train.eval()) if y==i or y==j])           
            y_ = ([y for (x, y) in zip(x_train.get_value(), y_train.eval()) if y==i or y==j])
            x_transformed = pca_(x_)

            plots[i, j].scatter(x_transformed[:, 0], x_transformed[:, 1], c=y_)
            plots[i, j].set_xticks(())
            plots[i, j].set_yticks(())

            if i == 0:
                plots[i, j].set_title(j)
                plots[j, i].set_ylabel(j)
    plt.tight_layout()
    plt.savefig(dataset + '_theano.png')

            
def plot_cifar():
    f = open('cifar-10/data_batch_1', 'rb')
    d = cPickle.load(f)
    f.close()

    data_x = d['data']
    data_y = d['labels']
    
    x_train = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
                             borrow=True)
    y_train = T.cast(theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                                   borrow=True), 'int32')
    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    
    for i in range(10):
        for j in range(10):
            print(i,j)
            x_ = numpy.asarray([x for (x, y) in zip(x_train.get_value(), y_train.eval()) if y==i or y==j])           
            y_ = ([y for (x, y) in zip(x_train.get_value(), y_train.eval()) if y==i or y==j])
            x_transformed = pca_(x_)

            plots[i, j].scatter(x_transformed[:, 0], x_transformed[:, 1], c=y_)
            plots[i, j].set_xticks(())
            plots[i, j].set_yticks(())

            if i == 0:
                plots[i, j].set_title(j)
                plots[j, i].set_ylabel(j)

    plt.tight_layout()
    plt.savefig('cifar_theano.png')
    
    
def scatterplot(dataset='mnist.pkl.gz'):

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X_train, y_train = train_set
    
    pca = RandomizedPCA(n_components=2)
    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue
        X_ = X_train[(y_train == i) + (y_train == j)]
        y_ = y_train[(y_train == i) + (y_train == j)]
        X_transformed = pca.fit_transform(X_)
        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())
      
        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title(j)
            plots[j, i].set_ylabel(j)
        
        #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plt.tight_layout()
    plt.savefig("mnist_pairs.png")

if __name__ == '__main__':
    #pca()
    #scatterplot()
    #plot_class()
    plot_cifar()
