# 2015-DL-BastienAchard
Deep learning for the real world 2015

Instructions:
=============

1. place the datasets (mnist.pkl.gz and cifar-10-python.tar.gz) in the folder "datasets", and extract the cifar dataset. The "datasets" folder should look like:
./datasets/
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
├── cifar-10-python.tar.gz
├── mnist.pkl.gz
└── put datasets here.txt

2. please execute the python scripts directly in their folder, e.g.
$ cd logreg
$ python logreg.py
for their are relative imports between files

3. For t-SNE, please place the bhtsne python implementation provided at http://lvdmaaten.github.io/tsne in the folder tsne/bhtsne

4. Python packages theano, climin matplotlib, cPickle and gzip are required

Questions:
==========

2. Multiclass logistic regression
---------------------------------
* LogisticRegression - file: logreg/logreg.py - usage: python logreg.py

P8: To implemented multiclass logistic regression, I first read and follow the tutorial at http://deeplearning.net/tutorial/logreg.html to get the general idea of how to implement the logistic regression class using theano. 
Then I adapted the code to use it with climin optimisers.
My implementation uses climin to create the minibatches and uses the climin gradient descent optimiser as initial optimisation method.
However it is way slower than the implementation from http://deeplearning.net/tutorial/logreg.html

P9: Using the implementation from the tutorial, we can achieve an error rate on the test set of about 7.5%. Using my implementation with similar parameters, I can achieve an error rate on the test set of about 7.8%.

P10: I used different climin optimisers (Gradient Descent, RMSProp, Lbfgs, Nonlinear Conjugate Gradient)

P11: The receptive fields ...

P12: To avoid overfitting, we stop training when the validation error increases or does not decrease enough at some point. To implement that, I took inspiration from the early stopping mechanism proposed at http://deeplearning.net/tutorial/gettingstarted.html#opt-early-stopping.

P13:

Bonus question: It is a bad scientific practice because we are trying to tune the parameters of the classifier, and even modify the dataset, in order to reach a certain error rate on the test set. It does provide any improvement, and can be seen as cheating on the actual performance of the classifier.


3. Two-layer neural network
---------------------------
* MLP - file: nn/mlp.py - usage: python mlp.py

P14: To implement a neural network with one hidden layer, I first read and follow the tutorial at http://deeplearning.net/tutorial/mlp.html to get the general idea of how to implement it using theano.
Then I adapted the code to use it with climin optimisers.
My implementation makes it possible to use different optimisation algorithms. It uses climin to create the minibatches, uses climin gradient descent and rmsprop optimisers. L1 and L2 regularisation are also available. 

P15: 

P16: 

P17:

P18:

P19:



4. PCA and sparse autoencoder
-----------------------------

* PCA - file: latent/pca.py - usage: python pca.py

P20: I implemented PCA using Theano and the SVD decomposition of the centered input (instead of computing the eigenpairs of the covariance matrix of the input)

P21: PCA scatterplot on MNIST shows that most pairs of digits can be easily classified using the 2 principal components. Exception are pairs (3,5), (4,9), (5,8), (7,9), and to a lesser extent (2,3). Indeed, the digits in each of these pairs present many similarities.

* Sparse autoencoder - file: latent/sparse_autoencoder.py - usage: python sparse_autoencoder.py

P22: To implement the sparse autoencoder, I read the course at http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity. 
I used climin gradient descent optimiser for training. 
Several loss functions are available, L(x) = ||f(x)-x||_2^2, L_sparse(x) = L(x) + lambda*|h(x)|_1, and L_KL(x) = L(x) + beta*KL(rho)

P23:

P24:

P25:

P26: Sparse encoding on MNIST will only keep the important features of the digit, that is keeping the shape and getting rid of the thickness of the line. Indeed, we can see that phenomena on the reconstruction plot of the first 100 digits. This seems relevant as only the shape is important to recognize a digit, not the thickness of the line.

Bonus problem: see P22

5. t-SNE
--------

P29: I used Barnes-Hut implementation with Python wrapper provided at http://lvdmaaten.github.io/tsne to reproduce Figure 5 of the Barnes-Hut-SNE paper, as it is way faster than the t-SNE Python implementation.


6. k-Means
----------

P30: To implement k-Means I followed the paper by Adam Coates.

P31:


