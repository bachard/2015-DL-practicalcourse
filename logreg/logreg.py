import cPickle
import gzip
import numpy
import theano

from theano import tensor as T

import climin
import climin.util

import itertools

from matplotlib import pyplot as plt

import copy
# We import the utils file
import sys
sys.path.append("..")
import utils


class LogisticRegression(object):
    """ Multiclass Logistic Regression class
    """
    def __init__(self, input, n_in, n_out, params=None):
        """
        :param input: input (theano.tensor.matrix)
        :param n_in: number of dimensions of the input
        :param n_out: number of classes
        
        :attribute W: theano representation of matrix of weights
        :attribute b: theano representation of bias vector
        :attribute params: theano representation of vector containing 
        the flatten representation of W and b; can be passed to the 
        object (useful to connect it in a MLP)

        :attribute W_values: array containing the actual values of W
        :attribute b_values: array containing the actual values of b
        :attribute params_values: flatten representation of W_values and b_values,
        useful for climin

        :function p_y_given_x: theano function returning the value of y given x
        :function y_pred: theano function returning the prediction for y given x
        """
        
        self.n_in = n_in
        self.n_out = n_out
        
        if params:
            self.params = params
        else:
            self.params = T.vector()
            
        self.W = self.params[:self.n_in * self.n_out].reshape((self.n_in, self.n_out))
        self.b = self.params[self.n_in * self.n_out:].reshape((self.n_out, ))

        self.W_values = numpy.zeros((self.n_in, self.n_out), dtype=theano.config.floatX)
        self.b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.params_values = numpy.concatenate([self.W_values.flatten(), self.b_values.flatten()])
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


    def negative_loglikelihood(self, y):
        """Theano function computing the negative loglikelihood for input x and target y """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Theano function computing the number of prediction errors in the classifier"""
        return T.mean(T.neq(self.y_pred, y))

    def set_values(self):
        """Sets the values of matrix of weights W and bias vector b contained in params_values"""
        self.W_values = self.params_values[:self.n_in * self.n_out].reshape((self.n_in, self.n_out))
        self.b_values = self.params_values[self.n_in * self.n_out:].reshape((self.n_out,))
    
    def get_values(self):
        """Returns the values of W and b"""
        return (self.W_values, self.b_values)
    


def train_logreg(dataset='../datasets/mnist.pkl.gz', n_in=28*28, n_out=10, optimizer="GradientDescent", learning_rate=0.1, momentum=0.1, batch_size=200, max_iter=40):
    """
    Trains a MLP with one hidden layer on a dataset (default MNIST) using climin optimizers

    :param dataset: path to the dataset (pkl.gz file), containing train_set, valid_set and test_set
    :param n_in: number of dimensions of the input
    :param n_out: number of output classes
    :param optimizer: climin optimizer to use (GradientDescent, RmsProp, Lbfgs, NonlinearConjugateGradient)
    :param learning_rate: learning rate of the optimizer
    :param momentum: momentum of the optimizer (only gradient descent)
    :param batch_size: size of a mini batch
    :param max_iter: max number of iterations for the optimizer (number of passes)
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

    # we construct the Logistic Regression classifier
    classifier = LogisticRegression(x, n_in, n_out)

    # symbolic cost function to be optimized
    cost = classifier.negative_loglikelihood(y)
    # errors of the mlp
    errors = classifier.errors(y)

    # we define the gradients of cost wrt the parameters of the classifier
    g_W = T.grad(cost, classifier.W)
    g_b = T.grad(cost, classifier.b)
    g_params = T.concatenate([g_W.flatten(), g_b.flatten()])

    # we define the functions to be passed to the climin optimizer
    f_errors = theano.function([classifier.params, x, y], errors)
    f_cost = theano.function([classifier.params, x, y], cost)
    f_g_params = theano.function([classifier.params, x, y], g_params)

    # number of iterations to process all the input set once
    pass_size = x_train.shape[0] / batch_size
    
    # we define different climin optimizers     
    opt_gd = climin.GradientDescent(
        wrt=classifier.params_values,
        fprime=f_g_params,
        step_rate=learning_rate,
        momentum=momentum,
        momentum_type="standard",
        args=args
    )

    opt_rmsprop = climin.RmsProp(
        wrt=classifier.params_values,
        fprime=f_g_params,
        step_rate=learning_rate,
        args=args
    )

    opt_lbfgs = climin.Lbfgs(
        wrt=classifier.params_values,
        f=f_cost,
        fprime=f_g_params,
        args=args
    )
    
    opt_nlcg = climin.NonlinearConjugateGradient(
        wrt=classifier.params_values,
        f=f_cost,
        fprime=f_g_params,
        args=args
    )

    if optimizer == "GradientDescent":
        opt = opt_gd
    elif optimizer == "RmsProp":
        opt = opt_rmsprop
    elif optimizer == "Lbfgs":
        opt = opt_lbfgs
    elif optimizer == "NonlinearConjugateGradient":
        opt = opt_nlcg
    else:
        print("Optimizer unknown, using GradientDescent (optimizers available: GradientDescent, RmsProp, Lbfgs, NonlinearConjugateGradient)")
        opt = opt_gd

    # we define variables to store the various errors  
    stopping_criteria = 0.0001
    train_errors = []
    valid_errors = []
    test_errors = []

    patience = 2 * pass_size
    patience_increase = 2
    
    train_error = None
    valid_error = None
    test_error = None
    
    best_params = None
    improvement_threshold = 0.995
    best_valid_error = numpy.inf
    n_iter_best = 0

    n_pass = 0
    
    for info in opt:
        n_iter = info['n_iter']
        previous_valid_error = valid_error
        train_error = f_errors(classifier.params_values, x_train, y_train) * 100
        valid_error = f_errors(classifier.params_values, x_valid, y_valid) * 100
        test_error = f_errors(classifier.params_values, x_test, y_test) * 100

        # we display information
        if n_iter % pass_size == 0 or n_iter == 1:
            print("Errors at iteration {} (pass {}): training set {} %, validation set {} %, test set {} %".format(
                n_iter,
                n_pass,
                train_error,
                valid_error,
                test_error)
            )

            n_pass += 1
            
            if valid_error < best_valid_error:
                if valid_error < best_valid_error * improvement_threshold:
                    patience = max(patience, n_iter + patience_increase * pass_size)
                best_valid_error = valid_error
                best_params = copy.deepcopy(classifier.params_values)
                n_iter_best = n_iter
                
        train_errors.append(train_error)
        valid_errors.append(valid_error)
        test_errors.append(test_error)
        
        if n_iter > patience:
            break
        
        if n_iter >= max_iter * pass_size:
            break
        

    train_errors = train_errors[:n_iter_best]
    test_errors = test_errors[:n_iter_best]
    valid_errors = valid_errors[:n_iter_best]
        
    print("Errors at final iteration: training set {} %, validation set {} %, test set {} %".format(
        train_errors[-1],
        valid_errors[-1],
        test_errors[-1])
    )
    
    print("Minimum test error achieved {} %".format(min(test_errors)))

    n_iters = len(train_errors)

    classifier.params_values = best_params
    
    classifier.set_values()
    
    print("Saving receptive fields to repflds.png...")
    utils.visualize_matrix(classifier.W_values.T, 10, 28, "repflds.png", cmap="gray_r", dpi=150)

    print("Plotting errors to errors.png...")
    iters = numpy.arange(n_iters)
    plt.clf()
    plt.plot(iters, train_errors, 'b', iters, valid_errors, 'g', iters, test_errors, 'r')
    plt.savefig("error.png")



if __name__ == "__main__":
    train_logreg(
        optimizer="GradientDescent",
        learning_rate=0.1,
        momentum=0.05,
        batch_size=100,
        max_iter=40
    )
