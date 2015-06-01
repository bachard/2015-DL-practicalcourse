# -*- coding: utf-8 -*-

import theano
from theano import tensor as T
import numpy
import climin
import gzip
import cPickle

import climin
import climin.util

from matplotlib import pyplot as plt

import time
# We import the utils file
import sys
sys.path.append("..")
import utils


class SparseAutoencoder(object):
    """ Sparse autoencoder class
    sparsity can be enforced using a L1 penalty of a KL-divergence penalty
    (see http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity)
    """
    def __init__(self, input, n_in, n_hidden, lambda_=0.05, sparsity_parameter=0.05, beta=0.05):
        """
        :param input: input (theano.tensor.matrix)
        :param n_in: number of dimensions of the input
        :param n_hidden: number of hidden units
        :param lambda_: L1 penalty parameter
        :param sparsity_parameter: sparsity parameter for KL-divergence penalty
        :param beta: KL-divergence penalty parameter

        :attribute W: theano representation of matrix of weights of hidden layer
        :attribute b: theano representation of bias vector of hidden layer
        :attribute W_prime: theano representation of matrix of weights of output layer;
        here we use tied weights thus W_prime = W.T
        :attribute b: theano representation of bias vector of output layer
        :attribute params: theano representation of vector containing 
        the flatten representation of W, b, b_prime

        :attribute W_values: array containing the actual values of W
        :attribute b_values: array containing the actual values of b
        :attribute b_prime_values: array containing the actual values of b
        :attribute params_values: flatten representation of W_values and b_values,
        useful for climin

        :function activation_hl: theano function returning the activation of the hidden layer
        :function activation: theano function returning the activation of the output layer
        :function activation_hl: theano function returning the activation of the hidden layer
        :function average_activation: theano function returning the average activation of the hidden layer
        :function cost: theano function returning the cost function without sparsity enforced
        :function cost_L1: theano function returning the cost function with L1 penalty
        :function cost_KL: theano function returning the cost function with KL-divergence penalty
        """     

        rng = numpy.random.RandomState(1211)

        self.n_in = n_in
        self.n_hidden = n_hidden
        
        self.params = T.vector()
        
        self.W = self.params[:n_in*n_hidden].reshape((n_in, n_hidden))

        self.b = self.params[n_in*n_hidden:n_in*n_hidden + n_hidden].reshape((n_hidden,))

        self.b_prime = self.params[n_in*n_hidden + n_hidden:].reshape((n_in,))
        self.W_prime = self.W.T

        self.params_list = [self.W, self.b, self.b_prime]

        # we initialize the weight matrix values which are uniformly
        # sampled from 4 *sqrt(-6./(n_in+n_hidden)) and 4 * sqrt(6./(n_in+n_hidden))
        # for sigmoid activation function;
        # (according to MLP tutorial on https://http://deeplearning.net/tutorial/mlp.html)
        self.W_values = numpy.asarray(
            rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_in + n_hidden)),
                high=4 * numpy.sqrt(6. / (n_in + n_hidden)),
                size=(n_in, n_hidden)
            ),
            dtype=theano.config.floatX
        )

        self.b_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)

        self.b_prime_values = numpy.zeros((n_in,), dtype=theano.config.floatX)

        self.params_values = numpy.concatenate([self.W_values.flatten(), self.b_values.flatten(), self.b_prime_values.flatten()])
        
        self.x = input

        self.rho = sparsity_parameter

        self.beta = beta

        self.activation_hl = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        self.activation = T.nnet.sigmoid(T.dot(self.activation_hl, self.W_prime) + self.b_prime)
        
        self.average_activation = T.mean(self.activation_hl, axis=1)
        
        self.sum_KL_divergence = T.sum(self.KL_divergence(self.rho, self.average_activation))

        # cost = sum_{i=1}^m ||x_i - z_i||^2
        self.cost = T.mean((self.x-self.activation).norm(L=2, axis=1)**2)
        # cost with L1 regularisation parameter
        self.cost_L1 = self.cost + lambda_ * T.mean((self.activation_hl).norm(L=1, axis=1))
        # cost with KL regularisation parameter
        self.cost_KL = self.cost + self.beta*self.sum_KL_divergence

        
    def KL_divergence(self, p, q):
        """ Compute the KL-divergence for probabilities p and q """
        return p * (T.log(p) - T.log(q)) + (1 - p) * (T.log(1 - p) - T.log(1 - q))
        
    def set_values(self):
        """Sets the values of matrix of weights W and bias vectors b and b_prime contained in params_values"""
        self.W_values = self.params_values[:self.n_in*self.n_hidden].reshape((self.n_in, self.n_hidden))
        self.b_values = self.params_values[self.n_in*self.n_hidden:self.n_in*self.n_hidden + self.n_hidden].reshape((self.n_hidden,))
        self.b_prime_values = self.params_values[self.n_in*self.n_hidden + self.n_hidden:].reshape((self.n_in,))




        
def train_SPA(dataset="../datasets/mnist.pkl.gz", n_in=28*28, n_hidden=5, learning_rate=0.1, momentum=0.0, cost_param=None, lambda_=0.0, sparsity_parameter=0.05, beta=0.1, batch_size=200, max_iters=15):
    """
    Trains a sparse entoencoder on a dataset (default MNIST) using climin optimizers

    :param dataset: path to the dataset (pkl.gz file), containing train_set, valid_set and test_set
    :param n_in: number of dimensions of the input
    :param n_hidden: number of hidden units
    :param learning_rate: learning rate of the gradient descent optimizer
    :param momentum: momentum of the gradient descent optimizer
    :param cost_param: type of penalty to enforce sparsity (None, "L1" or "KL")
    :param lambda_: L1 penalty parameter
    :param sparsity_parameter: sparsity parameter for KL-divergence penalty
    :param beta: KL-divergence penalty parameter
    :param batch_size: size of a mini batch
    :param max_iters: max number of iterations for the optimizer (number of passes)
    """
    

    # We load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
       
    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    x_test, y_test = test_set

    y_train = y_train.astype(dtype='int32')
    y_valid = y_valid.astype(dtype='int32')
    y_test = y_test.astype(dtype='int32')

    # we construct the minibatches sequence to be passed to the climin optimizer
    args = ((i, {}) for i in climin.util.iter_minibatches([x_train], batch_size, [0]))

    # theano symbolic representations of x (input)
    x = T.matrix('x')

    spa = SparseAutoencoder(
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        lambda_=lambda_,
        sparsity_parameter=0.01,
        beta=0.5
    )

    if cost_param == "L1":
        cost = spa.cost_L1
    elif cost_param == "KL":
        cost = spa.cost_KL
    else:
        cost = spa.cost
        
    g_params = T.concatenate([T.grad(cost, param).flatten() for param in spa.params_list])
    
    f_cost = theano.function([spa.params, x], cost)
    f_g_params = theano.function([spa.params, x], g_params)
    f_output = theano.function([spa.params, x], spa.activation)

    # number of iterations to process all the input set once
    pass_size = x_train.shape[0] / batch_size

    # climin Gradient Descent optimizer
    opt_gd = climin.GradientDescent(
        wrt=spa.params_values,
        fprime=f_g_params,
        step_rate=learning_rate,
        momentum=momentum,
        momentum_type="standard",
        args=args
    )

    n_pass = 0
    print("Optimisation using gradient descent...")
    for info in opt_gd:
        n_iter = info['n_iter']
        # we display information
        if n_iter % pass_size == 0 or n_iter == 1:
            print("Cost at pass {}: {}".format(n_pass, f_cost(spa.params_values, x_train)))
            n_pass += 1
        if n_iter >= max_iters*pass_size:
            break
    
    spa.set_values()
    output_100 = f_output(spa.params_values, x_train[0:100])

    prefix = "{}_{}_HU_{}".format(cost_param, lambda_, n_hidden)
    print("Saving SPA reconstruction errors on first 100 MNIST digits to autoencodererr_{}.png...".format(prefix))
    utils.visualize_matrix(output_100, 100, 28, "autoencodererr_{}.png".format(prefix), cmap="gray_r", dpi=300)

    # Visualization
    pixels = spa.W_values / numpy.linalg.norm(spa.W_values, 2, axis=0)
    print("Saving SPA filters visualization to autoencoderfilter_{}.png...".format(prefix))
    utils.visualize_matrix(pixels.T, n_hidden, 28, "autoencoderfilter_{}.png".format(prefix), cmap="gray_r", dpi=300)
                    

if __name__ == '__main__':
    start = time.time()
    train_SPA(
        n_hidden=28*28,
        learning_rate=0.1,
        momentum=0.01,
        cost_param="L1",
        lambda_=0.1,
        sparsity_parameter=0.05,
        beta=0.1,
        batch_size=20,
        max_iters=15
    )
        
    end = time.time()
    print("Total running time: {}s".format(end-start))
