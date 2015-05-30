import cPickle
import gzip
import numpy
import theano

from theano import tensor as T

import climin
import climin.util

import itertools

from matplotlib import pyplot as plt

# We import the utils file
import sys
sys.path.append("..")
import utils



class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        self.n_in = n_in
        self.n_out = n_out

        self.params = T.vector('params')
        self.W = self.params[:self.n_in * self.n_out].reshape((self.n_in, self.n_out))
        self.b = self.params[self.n_in * self.n_out:].reshape((self.n_out, ))

        self.W_values = numpy.zeros((self.n_in, self.n_out), dtype=theano.config.floatX)
        self.b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.params_values = numpy.concatenate([self.W_values.flatten(), self.b_values.flatten()])
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


    def negative_loglikelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

    def set_values(self):
        self.W_values = self.params_values[:self.n_in * self.n_out].reshape((self.n_in, self.n_out))
        self.b_values = self.params_values[self.n_in * self.n_out:].reshape((self.n_out,))
    
    def get_values(self):
        return (self.W_values, self.b_values)
    


def optimization(dataset='mnist.pkl.gz', batch_size=100):

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
       
    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    x_test, y_test = test_set

    y_train = y_train.astype(dtype='int32')
    y_valid = y_valid.astype(dtype='int32')
    y_test = y_test.astype(dtype='int32')
    
    args = ((i, {}) for i in climin.util.iter_minibatches([x_train, y_train], batch_size, [0, 0]))

    x = T.matrix('x')
    y = T.ivector('y')

    n_in = 28 * 28
    n_out = 10
    
    classifier = LogisticRegression(x, n_in, n_out)
    
    cost = classifier.negative_loglikelihood(y)
    errors = classifier.errors(y)
    g_W = T.grad(cost, classifier.W)
    g_b = T.grad(cost, classifier.b)
    g_params = T.concatenate([g_W.flatten(), g_b.flatten()])

    f_errors = theano.function([classifier.params, x, y], errors)
    f_cost = theano.function([classifier.params, x, y], cost)
    f_g_params = theano.function([classifier.params, x, y], g_params)

        
    opt_gd = climin.GradientDescent(
        wrt=classifier.params_values,
        fprime=f_g_params,
        step_rate=0.1,
        momentum=0.1,
        momentum_type="standard",
        args=args
    )

    opt_rmsprop = climin.RmsProp(
        wrt=classifier.params_values,
        fprime=f_g_params,
        step_rate=0.01,
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

    opt = opt_nlcg


    stopping_criteria = 0.0001
    train_errors = []
    valid_errors = []
    test_errors = []
    previous_valid_error = None
    train_error = None
    valid_error = None
    test_error = None
    
    for info in opt:
        n_iter = info['n_iter']
        previous_valid_error = valid_error
        train_error = f_errors(classifier.params_values, x_train, y_train) * 100
        valid_error = f_errors(classifier.params_values, x_valid, y_valid) * 100
        test_error = f_errors(classifier.params_values, x_test, y_test) * 100

        if n_iter % 100 == 0 or n_iter == 1:
            print("Errors at iteration {}: training set {} %, validation set {} %, test set {} %".format(
                n_iter,
                train_error,
                valid_error,
                test_error)
            )
        
        train_errors.append(train_error)
        valid_errors.append(valid_error)
        test_errors.append(test_error)
        # if not previous_valid_error is None:
        #     delta = previous_valid_error - valid_error
        #     if delta < stopping_criteria:
        #         print("Stopping criteria was reached at iteration {}...".format(n_iter))
        if n_iter >= 1000:
            break
        if test_error < 7:
            break

    print("Errors at final iteration: training set {} %, validation set {} %, test set {} %".format(
        train_error,
        valid_error,
        test_error)
    )
    print("Minimum test error achieved {} %".format(min(test_errors)))
    n_iter = len(train_errors)

    classifier.set_values()
    
    print("Saving receptive fields to repflds.png...")
    utils.visualize_matrix(classifier.W_values.T, 1, 10, 28, "repflds.png", cmap="gray_r")

    print("Plotting errors to errors.png...")
    iters = numpy.arange(n_iter)
    plt.clf()
    plt.plot(iters, train_errors, 'b', iters, valid_errors, 'g', iters, test_errors, 'r')
    plt.savefig("error.png")




if __name__ == "__main__":
    optimization()
