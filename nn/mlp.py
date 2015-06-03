import theano
import theano.tensor as T
import numpy

import climin
import climin.util
import climin.schedule

import gzip
import cPickle

from matplotlib import pyplot as plt

import copy
import time
# We import the utils file and LogisticRegression class
import sys
sys.path.append("..")
from logreg.logreg import LogisticRegression
import utils

class HiddenLayer(object):
    """ Hidden layer class used for multi layer perceptrons
    """

    def __init__(self, input, n_in, n_out, params=None, activation=T.tanh):
        """
        :param input: input (theano.tensor.matrix)
        :param n_in: number of dimensions of the input
        :param n_out: number of dimensions of the output
        :param activation: activation function used to compute the output
        
        :attribute rng: random generator
        :attribute W: theano representation of matrix of weights
        :attribute b: theano representation of bias vector
        :attribute params: theano representation of vector containing 
        the flatten representation of W and b; can be passed to the 
        object (useful to connect it in a MLP)

        :attribute W_values: array containing the actual values of W
        :attribute b_values: array containing the actual values of b
        :attribute params_values: flatten representation of W_values and b_values,
        useful for climin

        :function output: theano function returning the output of the hidden layer
        """       

        self.input = input

        # data preprocessing idea for tanh activation function
        # does not give better results
        # if activation == theano.tensor.tanh:
        #     self.input = input - T.mean(input, axis=0)
        
        self.rng = numpy.random.RandomState(1234)
        self.n_in = n_in
        self.n_out = n_out

        if params:
            self.params = params
        else:
            self.params = T.vector()
            
        self.W = self.params[:self.n_in * self.n_out].reshape((self.n_in, self.n_out))
        self.b = self.params[self.n_in * self.n_out:].reshape((self.n_out, ))

        # we initialize the weight matrix values which are uniformly
        # sampled from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function;
        # for sigmoid activation we multiply values by 4
        # (according to MLP tutorial on https://http://deeplearning.net/tutorial/mlp.html)
        # for softplus we break the symmetry and respread the values

        if activation == theano.tensor.nnet.softplus:
            self.W_values = numpy.asarray(
                self.rng.uniform(
                    low=-4*numpy.sqrt(6. / (n_in + n_out)),
                    high=0.25*numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

        else:        
            self.W_values = numpy.asarray(
                self.rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

        if activation == theano.tensor.nnet.sigmoid:
            self.W_values *= 4
        
        self.b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

        self.params_values = numpy.concatenate([self.W_values.flatten(), self.b_values.flatten()])
                
        lin_output = T.dot(self.input, self.W) + self.b

        if activation is None:
            self.output = lin_output
        else:
            self.output = activation(lin_output)


    def set_values(self):
        """Sets the values of matrix of weights W and bias vector b contained in params_values""" 
        self.W_values = self.params_values[:self.n_in * self.n_out].reshape((self.n_in, self.n_out))
        self.b_values = self.params_values[self.n_in * self.n_out:].reshape((self.n_out,))
        

class MLP(object):
    """ Multi layer perceptron class with one hidden layer
    """
    def __init__(self, input, n_in, n_hidden, n_out, activation=T.tanh):
        """
        :param input: input (theano.tensor.matrix)
        :param n_in: number of dimensions of the input
        :param n_hidden: number of units of the hidden layer
        :param n_out: number of classes
        :param activation: activation function of the hidden layer
        
        :attribute hiddenLayer: hidden layer of the mlp
        :attribute logRegLayer: output layer of the mlp
        :attribute params: theano representation of vector containing 
        the flatten representation of params of hiddenLayer and logRegLayer

        :attribute params_values: flatten representation of param_values of 
        hiddenLayer and logRegLayer (useful for climin)

        :function L1: theano function returning the sum of L1 norm of the matrices 
        of weights of hiddenLayer and logRegLayer 
        :function L2: theano function returning the sum of L2 norm of the matrices 
        of weights of hiddenLayer and logRegLayer 
        :function negative_loglikelihood: negative_loglikelihood function of the 
        logRegLayer
        :function errors: errors function of the logRegLayer
        """
        
        self.params = T.vector()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        # theano symbolic vector to be passed to the hidden layer
        hl_params = self.params[:n_in*n_hidden + n_hidden]
        # theano symbolic vector to be passed to the output layer
        lrl_params = self.params[n_in*n_hidden + n_hidden:]

        self.hiddenLayer = HiddenLayer(input, n_in, n_hidden, params=hl_params, activation=activation)
        
        self.logRegLayer = LogisticRegression(self.hiddenLayer.output, n_hidden, n_out, params=lrl_params)
        
        self.params_values = numpy.concatenate([self.hiddenLayer.params_values, self.logRegLayer.params_values])
        
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegLayer.W).sum()

        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegLayer.W ** 2).sum()
        
        self.negative_loglikelihood = self.logRegLayer.negative_loglikelihood

        self.errors = self.logRegLayer.errors
        
    def set_values(self):
        """Sets the values of params_values of hiddenLayer and logRegLayer"""
        self.hiddenLayer.params_values = self.params_values[:self.n_in * self.n_hidden + self.n_hidden]
        self.logRegLayer.params_values = self.params_values[self.n_in * self.n_hidden + self.n_hidden:]
        self.hiddenLayer.set_values()
        self.logRegLayer.set_values()
        



def train_mlp(dataset="../datasets/mnist.pkl.gz", n_in=28*28, n_hidden=300, n_out=10, activation=T.tanh, L1_reg=0.0, L2_reg=0.0, learning_rate=0.1, momentum=0.0, learning_rate_rmsprop=0.1, momentum_rmsprop=0.0, batch_size=200, max_iters_gd=40, max_iters_rmsprop=5):
    """
    Trains a MLP with one hidden layer on a dataset (default MNIST) using climin optimizers
    First optimisation using RMSProp, then possibly continues optimisation with gradient descent

    :param dataset: path to the dataset (pkl.gz file), containing train_set, valid_set and test_set
    :param n_in: number of dimensions of the input
    :param n_hidden: number of hidden units
    :param n_out: number of output classes
    :param activation: activation function of the hidden layer
    :param L1_reg: L1 regularisation parameter for training the MLP
    :param L2_reg: L2 regularisation parameter for training the MLP
    :param learning_rate: learning rate of the gradient descent optimizer
    :param momentum: momentum of the gradient descent optimizer
    :param learning_rate_rmsprop: learning rate of the rmsprop optimizer
    :param momentum_rmsprop: momentum of the rmsprop optimizer
    :param batch_size: size of a mini batch
    :param max_iters_gd: max number of iterations for the GD optimizer (number of passes)
    :param max_iters_rmsprop: max number of iterations for the RMSPROP optimizer (number of passes)
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
    args = ((i, {}) for i in climin.util.iter_minibatches([x_train, y_train], batch_size, [0, 0]))

    # theano symbolic representations of x (input) and y (targets)
    x = T.matrix('x')
    y = T.ivector('y')

    # we construct the MLP with n_hidden hidden units on the hidden layer
    mlp = MLP(x, n_in, n_hidden, n_out, )

    # symbolic cost function to be optimized
    cost = mlp.negative_loglikelihood(y) + L1_reg*mlp.L1 + L2_reg*mlp.L2_sqr
    # errors of the mlp
    errors = mlp.errors(y)

    # we define the gradients of cost wrt the parameters of the mlp
    g_W_hl = T.grad(cost, mlp.hiddenLayer.W)
    g_b_hl = T.grad(cost, mlp.hiddenLayer.b)
    g_W = T.grad(cost, mlp.logRegLayer.W)
    g_b = T.grad(cost, mlp.logRegLayer.b)
    g_params = T.concatenate([g_W_hl.flatten(), g_b_hl.flatten(), g_W.flatten(), g_b.flatten()])

    # we define the functions to be passed to the climin optimizer
    f_errors = theano.function([mlp.params, x, y], errors)
    f_cost = theano.function([mlp.params, x, y], cost)
    f_g_params = theano.function([mlp.params, x, y], g_params)

    # step_rate_decay = climin.schedule.linear_annealing(0.2, 0.01, 100)
    # step_rate_decay = climin.schedule.decaying(0.2, 0.95)

    # number of iterations to process all the input set once
    pass_size = x_train.shape[0] / batch_size

    # climin Gradient Descent optimizer
    opt_gd = climin.GradientDescent(
        wrt=mlp.params_values,
        fprime=f_g_params,
        step_rate=learning_rate,
        momentum=momentum,
        momentum_type="standard",
        args=args
    )

    # climin RMSPROP optimizer
    opt_rmsprop = climin.RmsProp(
        wrt=mlp.params_values,
        fprime=f_g_params,
        step_rate=learning_rate_rmsprop,
        args=args
    )
    
    # we define variables to store the various errors  
    train_errors = []
    valid_errors = []
    test_errors = []
    previous_valid_error = None
    train_error = None
    valid_error = None
    test_error = None

    # early stopping parameters
    # inspired from http://deeplearning.net/tutorial/gettingstarted.html#opt-early-stopping
    patience = 2 * pass_size
    patience_increase = 2
    
    best_params = None
    improvement_threshold = 0.995
    best_valid_error = numpy.inf
    n_iter_best = 0

    # number of the actual pass
    n_pass = 0

    if max_iters_rmsprop > 0:
        print("Starting optimisation using rmsprop...")
        for info in opt_rmsprop:
            n_iter = info['n_iter']
            previous_valid_error = valid_error
            train_error = f_errors(mlp.params_values, x_train, y_train) * 100
            valid_error = f_errors(mlp.params_values, x_valid, y_valid) * 100
            test_error = f_errors(mlp.params_values, x_test, y_test) * 100
        
            train_errors.append(train_error)
            valid_errors.append(valid_error)
            test_errors.append(test_error)
        
            # at the end of each pass
            if n_iter % pass_size == 0 or n_iter == 1:
                print("Errors at iteration {} (pass {}): training set {} %, validation set {} %, test set {} %".format(
                    n_iter,
                    n_pass,
                    train_error,
                    valid_error,
                    test_error)
                )
                n_pass += 1
        
                # we check early stopping criteria 
                if valid_error < best_valid_error:
                    if valid_error < best_valid_error * improvement_threshold:
                        patience = max(patience, n_iter + patience_increase * pass_size)
        
                    best_valid_error = valid_error
                    best_params = copy.deepcopy(mlp.params_values)
                    n_iter_best = n_iter
                    
                print("    -- minimum validation error achieved {} %".format(min(valid_errors)))
        
            # we have reached early stopping criteria
            if n_iter > patience:
                break
        
            if n_iter >= max_iters_rmsprop * pass_size:
               break
        
        # we only keep errors up to the iteration where best error rate on
        # test set is achieve
        train_errors = train_errors[:n_iter_best-1]
        test_errors = test_errors[:n_iter_best-1]
        valid_errors = valid_errors[:n_iter_best-1]
    
    n_iters_rmsprop = n_iter_best

    # we reset patience parameter
    patience = 2 * pass_size
    if max_iters_gd > 0:
        print("Continuing optimisation using gradient descent...")
        for info in opt_gd:
            n_iter = info['n_iter']
            previous_valid_error = valid_error
            train_error = f_errors(mlp.params_values, x_train, y_train) * 100
            valid_error = f_errors(mlp.params_values, x_valid, y_valid) * 100
            test_error = f_errors(mlp.params_values, x_test, y_test) * 100
            
            train_errors.append(train_error)
            valid_errors.append(valid_error)
            test_errors.append(test_error)
            
            # at the end of each pass
            if n_iter % pass_size == 0 or n_iter == 1:
                print("Errors at iteration {} (pass {}): training set {} %, validation set {} %, test set {} %".format(
                    n_iter,
                    n_pass,
                    train_error,
                    valid_error,
                    test_error)
                )
                n_pass += 1
                
                # we check early stopping criteria 
                if valid_error < best_valid_error:
                    if valid_error < best_valid_error * improvement_threshold:
                        patience = max(patience, n_iter + patience_increase * pass_size)
                        
                    best_valid_error = valid_error
                    best_params = copy.deepcopy(mlp.params_values)
                    n_iter_best = n_iter + n_iters_rmsprop
                    
                print("    -- minimum validation error achieved {} %".format(min(valid_errors)))
                
            # we have reached early stopping criteria
            if n_iter > patience:
                break
            
            if n_iter >= max_iters_gd * pass_size:
                break
        
        # we only keep errors up to the iteration where best error rate on
        # test set is achieve
        train_errors = train_errors[:n_iter_best-1]
        test_errors = test_errors[:n_iter_best-1]
        valid_errors = valid_errors[:n_iter_best-1]

    print("Errors at final iteration: training set {} %, validation set {} %, test set {} %".format(
        train_error,
        valid_error,
        test_error)
    )
    print("Minimum validation error achieved {} %".format(min(valid_errors)))
    n_iters = len(train_errors)

    # we reset the params values with the best achieved
    mlp.params_values = best_params
    mlp.set_values()
    
    print("Saving receptive fields to repflds.png...")
    utils.visualize_matrix(mlp.hiddenLayer.W_values.T, n_hidden, 28, "repflds.png", cmap="gray", dpi=300)

    print("Plotting errors to errors.png...")
    iters = numpy.arange(n_iters)
    plt.clf()
    plt.plot(iters, train_errors, 'b', iters, valid_errors, 'g', iters, test_errors, 'r')
    plt.savefig("errors.png", dpi=300)
        
        
if __name__ == '__main__':
    start = time.time()
    train_mlp(
        n_hidden=300,
        activation=T.nnet.sigmoid,
        learning_rate=0.05,
        momentum=0.01,
        learning_rate_rmsprop=0.004,
        momentum_rmsprop=0.001,
        batch_size=100,
        max_iters_rmsprop=20,
        max_iters_gd=20
    )
    end = time.time()
    print("Total running time: {}s".format(end-start))
