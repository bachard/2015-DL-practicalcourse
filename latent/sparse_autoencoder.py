# -*- coding: utf-8 -*-

import theano
from theano import tensor as T
import numpy
import climin
import gzip
import cPickle

from matplotlib import pyplot as plt

class SparseAutoencoder(object):

    def __init__(self, input, n_in, n_hidden, sparsity_parameter=0.05, beta=0.05):


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

        self.activation_hl = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.activation = T.nnet.softmax(T.dot(self.activation_hl, self.W_prime) + self.b_prime)
        
        self.average_activation = T.mean(self.activation_hl, axis=1)
        
        self.sum_KL_divergence = T.sum(self.KL_divergence(self.rho, self.average_activation))

        # cost = sum_{i=1}^m ||x_i - z_i||^2 + beta*sum_KL_divergence
        self.cost = T.mean((self.x-self.activation).norm(L=2, axis=1)**2) + self.beta*self.sum_KL_divergence

        # Visualization
        self.pixels = self.W #/ self.W.norm(L=2, axis=0)

    def KL_divergence(self, p, q):
        return p * (T.log(p) - T.log(q)) + (1 - p) * (T.log(1 - p) - T.log(1 - q))
        
    # def average_activation(self):
    #         
    #     # Computes the matrix of activations in the hidden layer       
    #     # results, updates = theano.scan(
    #     #     lambda x_i: T.nnet.sigmoid(T.dot(self.W, x_i) + self.b),
    #     #     sequences=[X])
    #     # activation = theano.function(inputs=[], outputs=[results], givens={X: self.x})
    #     
    #     # Computes the average over the columns of the previous matrix
    #     # the result is a vector containing the average activation of each hidden unit
    #     # results, updates = theano.scan(lambda a_j: a_j.mean(), sequences=[self.activation_hl])
    #     # average_activation = results
    #     # return average_activation
    #     return T.mean(self.activation_hl, axis=1)


    # def sum_KL_divergence(self):
    #         """
    #         P: average activation vector
    #         """
    #         
    #         # results, updates = theano.scan(
    #         #     lambda p_j: self.rho * T.log(self.rho / p_j) + (1 - self.rho) * T.log ((1 - self.rho)/(1 - p_j)),
    #         #     sequences=[self.average_activation()])
    #         # return T.sum(results)
    #         return T.sum(self.KL_divergence(self.rho, self.average_activation()))

    # def get_output(self):
    #     # results, updates = theano.scan(
    #     #     lambda x_i: T.nnet.sigmoid(T.dot(self.W, x_i) + self.b),
    #     #     sequences=[X])
    #     # activation = theano.function(inputs=[], outputs=[results], givens={X: self.x})
    #     return T.nnet.softmax(T.dot(T.nnet.softmax(T.dot(input, self.W) + self.b)))
        
        
    # def get_cost(self):
    #     """
    #     cost = sum_{i=1}^m ||x_i - z_i||^2 + beta*sum_KL_divergence
    #     """
    #     return T.mean((self.x-self.activation).norm(2, axis=1)**2) + self.beta*self.sum_KL_divergence()

    
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


    
def test_SPA(dataset='mnist.pkl.gz', n_hidden=5, learning_rate=0.1, batch_size=20, training_epochs=15):

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
        sparsity_parameter=0.1,
        beta=0.1
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
    # for index in range(0, 100):
    #     if index <= n_epochs:
    #         print("Iteration n°{}".format(index))
    #         batch_cost = train_spa(index)
    #         print("Average batch cost: {}".format(batch_cost))
    #         print(f_g_params(index))
    #     else:
    #         break

    pixels = spa.pixels.eval()

    # for n in range(n_hidden):
    #     img = pixels[:, n].reshape(28,28)
    #     plt.imshow(img)
    #     plt.savefig('plots_spa/hidden_unit_{}.png'.format(n))

    n_h = 28
    n_w = 28
    
    fig, plots = plt.subplots(n_h, n_w)
    fig.set_size_inches(50, 50)
    plt.prism()
    
    
    for i in range(n_h):
        for j in range(n_w):
            print(i,j)
            plots[i, j].imshow(pixels[:, i+j].reshape((28, 28)))
            plots[i, j].set_xticks(())
            plots[i, j].set_yticks(())

            if i == 0:
                plots[i, j].set_title(j)
                plots[j, i].set_ylabel(j)
    plt.tight_layout()
    plt.savefig('plots/spa_plot.png')
    
    
    print(pixels.shape)
    print(spa.W.get_value().shape)
    print(spa.b.get_value().shape)
    print(spa.b_prime.get_value().shape)
            

if __name__ == '__main__':
    test_SPA(n_hidden=784, training_epochs=20)
        
        
