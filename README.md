# 2015-DL-BastienAchard
Deep learning for the real world 2015

Instructions:
=============

1. place the datasets (mnist.pkl.gz and cifar-10-python.tar.gz) in the folder "datasets", and extract the cifar dataset. The "datasets" folder should look like:

./datasets/
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
├── cifar-10-python.tar.gz
├── mnist.pkl.gz
└── put datasets here.txt

2. please execute the python scripts directly in their folder, e.g.
$ cd logreg
$ python logreg.py
for their are relative imports between files

3. For t-SNE, please place the bhtsne c++ implementation provided at http://lvdmaaten.github.io/tsne in the folder tsne/bhtsne, along with the python wrapper

4. Python packages theano, numpy, climin, matplotlib, scipy, cPickle and gzip are required

5. For each script (except t-SNE), you can modify the parameters directly in the file.

Questions:
==========

1. Remarks
----------
* In the file utils.py, I implemented a function to easily visualize receptive fields (values of matrix of weights) named visualize_matrix.

* To evaluate my implementations results on MNIST, I used the database provided at http://yann.lecun.com/exdb/mnist/

* I placed relevant plots for each task in the results folder of each subfolder.

2. Multiclass logistic regression
---------------------------------
* LogisticRegression - file: logreg/logreg.py - usage: python logreg.py

P8: To implemented multiclass logistic regression, I first read and follow the tutorial at http://deeplearning.net/tutorial/logreg.html to get the general idea of how to implement the logistic regression class using theano. 
Then I adapted the code to use it with climin optimisers.
My implementation uses climin to create the minibatches and uses the climin gradient descent optimiser as initial optimisation method.
However it is much slower than the implementation from http://deeplearning.net/tutorial/logreg.html

P9: Using the implementation from the tutorial, we can achieve an error rate on the test set of about 7.5%. Using my implementation with similar parameters (learning rate = 0.13, batch size = 600), I can achieve an error rate on the test set of about 7.8%.
According to the database results on MNIST at http://yann.lecun.com/exdb/mnist/, my results seems consistent compared to the results of linear classifiers without data preprocessing.

P10: I used different climin optimisers (Gradient Descent, RMSProp, Lbfgs, Nonlinear Conjugate Gradient) for this task.

P12: To avoid overfitting, we stop training when the validation error increases or does not decrease enough at some point. To implement that, I took inspiration from the early stopping mechanism proposed at http://deeplearning.net/tutorial/gettingstarted.html#opt-early-stopping.

P13: I could approach such an error rate with my implementation using RmsProp, learning rate=0.0035, momentum=0.0008, batch_size=100, giving me a minimum error rate on test set of 7.16%.

Bonus question: It is a bad scientific practice because we are trying to tune the classifier, and, in more problematic manner, modify the dataset, in order to reach a certain error rate on the test set. It does provide any improvement, and can be seen as cheating on the actual performance of the classifier.


3. Two-layer neural network
---------------------------
* MLP - file: nn/mlp.py - usage: python mlp.py

P14: To implement a neural network with one hidden layer, I first read and follow the tutorial at http://deeplearning.net/tutorial/mlp.html to get the general idea of how to implement it using theano.
Then I adapted the code to use it with climin optimisers.
My implementation makes it possible to use different optimisation algorithms. It uses climin to create the minibatches, uses climin gradient descent and rmsprop optimisers. L1 and L2 regularisation are also available. 
For early stopping, as in multiclass logistic regression implementation, I took inspiration from the early stopping mechanism proposed at http://deeplearning.net/tutorial/gettingstarted.html#opt-early-stopping.
Additionally, for the optimisation method, I had the idea of combining rmsprop with standard gradient descent, in sequential manner, that is we first optimise using rmsprop and then try to improve results by optmising with standard gradient descent. My idea was motivated by the fact that rmsprop is faster but variates more than standard gradient descent, thus I thought it was worth trying to finish optimisation using a more stable optimiser. You can use this feature by setting a non zero number of iterations for the gradient descent optimiser (you can tune learning rate and momentum parameters for both optimisers).

P15: Using 300 hidden units with tanh activation functions and rmsprop as the initial optimisation method, my implementation can achieve an error rate on the test set of about 2.5% (parameters: tanh activation function, learning rate=0.05, batch size=200, no momentum). Comparing with the database of results found at http://yann.lecun.com/exdb/mnist/, my results seems coherent.

P16: The file activation_functions.png is a plot of the following functions.
logistic sigmoid (theano.tensor.nnet.sigmoid, blue curve), varies from 0 to 1, very smooth at its inflection point 0.
tanh f(x) = (theano.tensor.tanh, green curve), varies from -1 to +1, steepest than sigmoid at its inflection point 0.
rectified linear neurons f(x)=max(0,x), using softplus function as approximation (we need the activation function to be differentiable) (theano.tensor.nnet.softplus, red curve), strictly increasing convex function, varies from 0 to +inf. The derivative of the softplus function is the sigmoid function.
These three functions do not have the same variation domain, smoothness nor same shape in the case of softplus, thus it will influence weights initialization. Say we choose to initialize weights for activation function tanh with values chosen uniformly at random in the range [-a:a] (the tanh function being symmetric). Then for activation function sigmoid, as the variation domain is smaller and the curve smoother, we should spread the initial weights even more, so choose a range like [-delta*a:delta*a], delta > 1.
According to the MLP tutorial (https://http://deeplearning.net/tutorial/mlp.html), we should initialize the weights with values chosen uniformly at random in the range [sqrt(-6./(n_in+n_hidden)):sqrt(6./(n_in+n_hidden))], and then choose choose delta=4 for sigmoid activation function.
The softplus function as a totally different shape compared to the two previous ones, as it is unbounded when values goes to infinity. Thus to prevent activation values to grow too much, we should break the symmetry and initialize the weights with values chosen uniformly at random in a range like [-delta*a:1/delta*a].
Testing my implement with choosing delta=4 for softplus activation function with the previous rule seems efficient.
Concerning data preprocessing, we can suppose that centering the data around 0 in the case of a tanh activation function could be a efficient as the tanh function is an odd function. However, implementing this data preprocessing idea does not give better results than without data preprocessing (thus I do not use it in the following). 

P19: Using RMSProp with tanh activation function, learning rate = 0.004,  momentum = 0.001, batch size=100, we can achieve an error rate on test set of about 2%, with minimal error rate on validation set of less than 1.8%.
Using RMSProp + GradientDescent with sigmoid activation function, learning_rate=0.05, momentum=0.01 for GD, learning_rate_rmsprop=0.004, momentum_rmsprop=0.001, batch_size=100, max_iters_rmsprop=20, max_iters_gd=20, we can achieve an error rate on test of 1.69%.



4. PCA and sparse autoencoder
-----------------------------

* PCA - file: latent/pca.py - usage: python pca.py

P20: I implemented PCA using Theano and the SVD decomposition of the centered input (instead of computing the eigenpairs of the covariance matrix of the input)

P21: PCA scatterplot on MNIST shows that most pairs of digits can be easily classified using the 2 principal components. Exception are pairs (3,5), (4,9), (5,8), (7,9), and to a lesser extent (2,3). Indeed, the two digits in each of these pairs present many similarities, thus only selecting the 2 first principal components is not enough to classify them.
PCA scatterplot on CIFAR-10 is not as conclusive as the previous one. Only a few classes seems to be separable using only the first 2 components. On the contrary to PCA scatterplot on MNIST, we can see than for the scatterplot for a single class, the points are much more spread. 

* Sparse autoencoder - file: latent/sparse_autoencoder.py - usage: python sparse_autoencoder.py

P22: To implement the sparse autoencoder, I read the course at http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity. 
I used climin gradient descent optimiser for training. 
Several loss functions are available, L(x) = ||f(x)-x||_2^2, L_sparse(x) = L(x) + lambda*|h(x)|_1, and L_KL(x) = L(x) + beta*KL(rho)

P23: see P22

P24: The higher lambda (L1 penalty parameter) is, the thiner and more blurred the digit are after reconstruction. A higher lambda will cause the matrix of weight to be sparser, and thus it will obviously yield more reconstruction errors.

P25: The higher lambda (L1 penalty parameter) is, the more the receptive fields tends to contain no relevant information. We can see on the plots (located in results), that if lambda is high, e.g. 1.0, most of the receptive fields will contain just noise, and a few will contain what seems to be average digits (I think it is comparable to the receptive fields Logistic Regression produces). If lambda is lower, say 0.5, more receptiv fields will contain information. This is logic as a high lambda will tend to make the weight matrix of the hidden layer sparser.

P26: Sparse encoding on MNIST will only keep the important features of the digit, that is keeping the shape and getting rid of the thickness of the line. Indeed, we can see that phenomena on the reconstruction plot of the first 100 digits. This seems relevant as only the shape is important to recognize a digit, not the thickness of the line.

Bonus problem: see P22

5. t-SNE
--------

* BH-tSNE - file: tsne/bhtsne_mnist.py - usage: python bhtnse_mnist NUM_SAMPLES d

P29: For this problem, I chose to use Barnes-Hut implementation with Python wrapper provided at http://lvdmaaten.github.io/tsne to reproduce Figure 5 of the Barnes-Hut-SNE paper. Initially I used the standard t-SNE Python implementation, but it is much slower than the t-SNE Python implementation, and requires much more memory space (and thus could not handle the entire MNIST set)
You can modify the parameter d which affect the spacing between digits, and thus how dense the figure will be (if d is high, say 3, the digits will be relatively far from each  other, if d is low, say 1, digits will be very close to each other).
Remarks: the script will save the results of bhtsne on the data in a pkl.gz file, so that you can modify d without having to recompute bhtsne on the data.


6. k-Means
----------

P30: To implement k-Means I followed the paper by Adam Coates. I also implemented a function to be able to correctly rescale the images from CIFAR-10 to any size in the RGB format.

P31: For this task I rescale the images from 32x32 to 12x12, and choose 500 centres.
Visualisation of the receptive fields is not very conclusive regarding the colors, as I think I have a problem rendering the RGB components.
Moreover on my machine (nvidia GTX 960), it takes about an hour for ten iterations on the entire CIFAR-10 set.


