"""  
    The vanilla implementation of GCN
    
    Copyright (c) H. Jin @ 2019-01-08
"""

import tensorflow as tf
import numpy as np
from models.utils import xavier_init, sp_matrix_to_sp_tensor

spdot = tf.sparse_tensor_dense_matmul
dot = tf.matmul
tf.set_random_seed(15)

class GCN():

    def __init__(self, sizes, A, X, param={}, with_relu=True):
        """ Initialize the GCN from A, X and Z

        Parameters
        ----------
        sizes: a tuple represent the decoder size
        A: a numpy.narray represents the adjacency matrix
        X: a numpy.narray represents the feature matrix
        param: a dictionary of optinal parameters

        Notes
        -----
        Initialization create the model, and the train function will create
            a sesssion later on.

        """
        self.A = sp_matrix_to_sp_tensor(A)
        self.X = sp_matrix_to_sp_tensor(X)

        # weight to learn
        self.W1 = tf.Variable(
                    xavier_init((X.shape[1], sizes[0])), dtype=tf.float32)
        self.W2 = tf.Variable(
                    xavier_init(sizes), dtype=tf.float32)
        
        self.node_ids = tf.placeholder(tf.int32, [None], 'node_ids')
        self.node_labels = tf.placeholder(tf.int32, [None, sizes[1]], 'node_labels')

        # build two layers
        act = tf.nn.relu if with_relu else lambda x: x
        self.h1 = self.SparseConv(self.A, self.X, act)
        self.h2 = self.DenseConv(self.A, self.h1)
        
        self.logits = tf.gather(self.h2, self.node_ids)
        self.predictions = tf.nn.softmax(self.logits)
        self.loss_per_node = tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=self.logits, 
                                labels=self.node_labels)

        self.loss = tf.reduce_mean(self.loss_per_node)

        ## operators
        self.opti = tf.train.AdamOptimizer(0.01).minimize(
                        self.loss, var_list=[self.W1, self.W2])
        self.dw1 = tf.gradients(self.loss, self.W1)[0]
        self.dw2 = tf.gradients(self.loss, self.W2)[0]

        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)

    def SparseConv(self, A, X, act=tf.nn.relu):
        """ Sparse tensor convolutional layer """
        _prod = spdot(X, self.W1)
        _prod = spdot(A, _prod)
        return act(_prod)

    def DenseConv(self, A, X, act=lambda x: x):
        """ Dense tensor convolutional layer """
        _prod = dot(X, self.W2)
        _prod = spdot(A, _prod)
        return act(_prod)
