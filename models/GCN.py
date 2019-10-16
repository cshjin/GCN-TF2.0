"""  
    The vanilla implementation of GCN
    
    Copyright (c) H. Jin @ 2019-01-08
"""

import tensorflow as tf
import numpy as np
from models.utils import xavier_init, sp_matrix_to_sp_tensor
from sklearn.metrics import accuracy_score

spdot = tf.sparse_tensor_dense_matmul
dot = tf.matmul
# tf.set_random_seed(15)

flags = tf.app.flags
FLAGS = flags.FLAGS


class GCN():

    def __init__(self, sizes, A, X, param={}, with_relu=True):
        """ Initialize the GCN from A, X and Z

        Parameters
        ----------
        sizes: a tuple represent the decoder size
        A: a numpy.narray represents the adjacency matrix
        X: a numpy.narray represents the feature matrix
        param: a dictionary of optinal parameters
        with_relu: indicator of using a nonlinear activation function

        Notes
        -----
        Initialization create the model, and the train function will create
            a sesssion later on.

        """
        # self.A1 = sp_matrix_to_sp_tensor(A1)
        # self.A2 = sp_matrix_to_sp_tensor(A2)
        self.A = sp_matrix_to_sp_tensor(A)
        self.X = sp_matrix_to_sp_tensor(X)

        # weight to learn
        self.W1 = tf.Variable(
            xavier_init((X.shape[1], sizes[0])), dtype=tf.float32)
        self.W2 = tf.Variable(
            xavier_init(sizes), dtype=tf.float32)
        dseq = A.sum(1).A1
        dseq = np.power(dseq, -0.5)
        self.agg_l = tf.Variable(dseq)
        self.agg_left = tf.diag(self.agg_l)
        self.agg_r = tf.Variable(dseq)
        self.agg_right = tf.diag(self.agg_r)

        self.A = spdot(self.A, self.agg_right)
        self.A = dot(self.agg_left, self.A)

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

        # operators
        self.opti = tf.train.AdamOptimizer(0.01).minimize(
            self.loss, var_list=[self.W1, self.W2, self.agg_l, self.agg_r])
        self.dw1 = tf.gradients(self.loss, self.W1)[0]
        self.dw2 = tf.gradients(self.loss, self.W2)[0]

        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)

    def SparseConv(self, A, X, act=tf.nn.relu):
        """ Sparse tensor convolutional layer 

        Parameters
        ----------
        A: sparse tensor
        X: sparse tensor
        act: activation function

        Returns
        -------
        A dense tensor
        """
        _prod = spdot(X, self.W1)
        _prod = dot(A, _prod)
        return act(_prod)

    def DenseConv(self, A, X, act=lambda x: x):
        """ Dense tensor convolutional layer 
        Parameters
        ----------
        A: sparse tensor
        X: dense tensor
        act: activation function

        Returns
        -------
        A dense tensor

        Notes
        -----
        The difference between the SparseConv and DenseConv is the type of X
        """
        _prod = dot(X, self.W2)
        _prod = dot(A, _prod)
        return act(_prod)

    def train_model(self, train_idx, val_idx, labels):
        """ Train the model with train set and validation set

        Parameters
        ----------
        train_idx:
        val_idx:
        labels:

        Returns
        -------
        None
        """
        feed_train = {self.node_ids: train_idx, self.node_labels: labels[train_idx]}
        feed_val = {self.node_ids: val_idx, self.node_labels: labels[val_idx]}
        best = 0
        threshold = 3
        early_stopping = threshold
        for epoch in range(1000):
            self.session.run(self.opti, feed_dict=feed_train)
            train_loss, train_acc = self.eval_model(train_idx, labels)
            val_loss, val_acc = self.eval_model(val_idx, labels)

            # early stopping
            if val_acc > best:
                best = val_acc
                early_stopping = threshold
            else:
                early_stopping -= 1
            if early_stopping == 0:
                break

            if FLAGS.verbose:
                print(
                    "epoch:{:03d}".format(epoch+1),
                    "train_loss:{:.3f}".format(train_loss),
                    "train_acc:{:.3f}".format(train_acc),
                    "val_loss:{:.3f}".format(val_loss),
                    "val_acc:{:.3f}".format(val_acc),
                )

    def eval_model(self, ids, labels):
        """ Evaluate the model by the given idx

        Parameters
        ----------
        idx:
        labels:

        Returns
        -------
        loss: 
        accuracy_rate:
        """
        feed_test = {self.node_ids: ids, self.node_labels: labels[ids]}
        preds, loss = self.session.run([self.predictions, self.loss], feed_dict=feed_test)
        pred_labels = preds.argmax(1)
        true_labels = labels.argmax(1)[ids]

        return loss, accuracy_score(true_labels, pred_labels)
