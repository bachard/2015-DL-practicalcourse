# -*- coding: utf-8 -*-

import theano
from theano import tensor as T
import numpy
import climin
import gzip
import cPickle

from matplotlib import pyplot as plt

# We import the utils file
import sys
sys.path.append("..")
import utils


class SparseAutoencoder(object):

    def __init__(self, input, n_in, n_hidden, lambda_=0.05, sparsity_parameter=0.05, beta=0.05):


        """ 
        Initial values for W, b, b'
        """
        rng = numpy.random.RandomState(1211)
        W_values = numpy.asarray(
            rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_in + n_hidden)),
                high=4 * numpy.sqrt(6. / (n_in + n_hidden)),
                size=(n_in, n_hidden)
            ),
            dtype=theano.config.floatX
        )

        self.W = theano.shared(value=W_values, name='W', borrow=True)

        self.b = theano.shared(value=numpy.zeros((n_hidden,), dtype=theano.config.floatX),
                          name='b',
                          borrow=True)

        self.b_prime = theano.shared(value=numpy.zeros((n_in,), dtype=theano.config.floatX),
                          name='b',
                          borrow=True)

        self.W_prime = self.W.T

        self.params = [self.W, self.b, self.b_prime]
        
        self.x = input

        self.rho = sparsity_parameter

        self.beta = beta

        self.activation_hl = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        self.activation = T.nnet.sigmoid(T.dot(self.activation_hl, self.W_prime) + self.b_prime)
        
        self.average_activation = T.mean(self.activation_hl, axis=1)
        
        self.sum_KL_divergence = T.sum(self.KL_divergence(self.rho, self.average_activation))

        # cost = sum_{i=1}^m ||x_i - z_i||^2 + beta*sum_KL_divergence
        self.cost = T.mean((self.x-self.activation).norm(L=2, axis=1)**2)

        self.cost_L1_pen = self.cost + lambda_ * T.mean((self.activation_hl).norm(L=1, axis=1))
        
        self.cost_KL = self.cost + self.beta*self.sum_KL_divergence

        # Visualization
        self.pixels = self.W / self.W.norm(L=2, axis=0)

    def KL_divergence(self, p, q):
        return p * (T.log(p) - T.log(q)) + (1 - p) * (T.log(1 - p) - T.log(1 - q))
        
    
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


def test_SPA(dataset="../datasets/mnist.pkl.gz", n_hidden=5, learning_rate=0.1, batch_size=20, training_epochs=15):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')

    n_in = 28 * 28
        
    spa = SparseAutoencoder(
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        lambda_=0.,
        sparsity_parameter=0.01,
        beta=0.5
    )

    cost = spa.cost
    g_params = [T.grad(cost, param) for param in spa.params]
    updates = [(param, param - learning_rate * g_param) for (param, g_param) in zip(spa.params, g_params)]
    
    f_cost = theano.function(
        inputs=[index],
        outputs=[cost],
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )

    f_g_params = theano.function(
        inputs=[index],
        outputs=g_params,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )

    train_spa = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )


    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_spa(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    pixels = spa.pixels.eval()

    f_output_100 = theano.function(
        inputs=[],
        outputs = spa.activation,
        givens={
            x: train_set_x[0: 100]
        }
    )

    output_100 = f_output_100()
    utils.visualize_matrix(output_100, 10, 10, 28, "autoencodererr.png", cmap="gray_r", dpi=300)
    
    utils.visualize_matrix(pixels.T, 28, 28, 28, "autoencoderfilter.png", cmap="gray_r", dpi=300)
    print(pixels.shape)
    print(spa.W.get_value().shape)
                

if __name__ == '__main__':
    test_SPA(n_hidden=28*28, training_epochs=40)
        
        
