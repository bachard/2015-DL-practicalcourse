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

5. For each script (except t-SNE), you can modify the parameters in the file.

Questions:
==========

2. Multiclass logistic regression
---------------------------------
* LogisticRegression - file: logreg/logreg.py - usage: python logreg.py

P8: To implemented multiclass logistic regression, I first read and follow the tutorial at http://deeplearning.net/tutorial/logreg.html to get the general idea of how to implement the logistic regression class using theano. 
Then I adapted the code to use it with climin optimisers.
My implementation uses climin to create the minibatches and uses the climin gradient descent optimiser as initial optimisation method.
However it is much slower than the implementation from http://deeplearning.net/tutorial/logreg.html

P9: Using the implementation from the tutorial, we can achieve an error rate on the test set of about 7.5%. Using my implementation with similar parameters (learning rate = 0.13, batch size = 600), I can achieve an error rate on the test set of about 7.8%.

P10: I used different climin optimisers (Gradient Descent, RMSProp, Lbfgs, Nonlinear Conjugate Gradient)

P12: To avoid overfitting, we stop training when the validation error increases or does not decrease enough at some point. To implement that, I took inspiration from the early stopping mechanism proposed at http://deeplearning.net/tutorial/gettingstarted.html#opt-early-stopping.

P13: 

Bonus question: It is a bad scientific practice because we are trying to tune the parameters of the classifier, and even modify the dataset, in order to reach a certain error rate on the test set. It does provide any improvement, and can be seen as cheating on the actual performance of the classifier.


3. Two-layer neural network
---------------------------
* MLP - file: nn/mlp.py - usage: python mlp.py

P14: To implement a neural network with one hidden layer, I first read and follow the tutorial at http://deeplearning.net/tutorial/mlp.html to get the general idea of how to implement it using theano.
Then I adapted the code to use it with climin optimisers.
My implementation makes it possible to use different optimisation algorithms. It uses climin to create the minibatches, uses climin gradient descent and rmsprop optimisers. L1 and L2 regularisation are also available. 
For early stopping, as in multiclass logistic regression implementation, I took inspiration from the early stopping mechanism proposed at http://deeplearning.net/tutorial/gettingstarted.html#opt-early-stopping.

P15: Using 300 hidden units with tanh activation functions and rmsprop as the initial optimisation method, my implementation can achieve an error rate on the test set of about 2.5% (parameters: learning rate=0.05, batch size=200)

P16: 

P19: Using RMSProp with learning rate = 0.004,  momentum = 0.001, batch size=100, we can achieve an error rate on test set of about 2%, with minimal error rate on validation set of less than 1.8%.



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

P29: I used Barnes-Hut implementation with Python wrapper provided at http://lvdmaaten.github.io/tsne to reproduce Figure 5 of the Barnes-Hut-SNE paper, as it is way faster than the t-SNE Python implementation, and requires less memory space.
You can modify the parameter d which affect the spacing between digits, and thus how dense the figure will be (if d is high, say 3, the digits will be relatively far from each  other, if d is low, say 1, digits will be very close to each other).
Remarks: the script will save the results of bhtsne on the data in a pkl.gz file, so that you can modify d without having to recompute bhtsne on the data.


6. k-Means
----------

P30: To implement k-Means I followed the paper by Adam Coates.

P31: For this task I rescale the images from 32x32 to 12x12, and choose 500 centres.
Visualisation of the receptive fields is not very conclusive regarding the colors, as I think I have a problem rendering the RGB components.


