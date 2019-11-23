###############################################################################
# TF 2.0 implementation of GCN.
# This is the version without using the keras API.
#
# Copyright (c) 2019, H. Jin
###############################################################################

import warnings
import os
import numpy as np
import scipy.sparse as sp
from utils import sp_matrix_to_sp_tensor
from sklearn.metrics import accuracy_score
from time import time

SEED = 15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

try:
    import tensorflow.compat.v2 as tf
    tf.random.set_seed(SEED)
    spdot = tf.sparse.sparse_dense_matmul
except ImportError:
    import tensorflow as tf
    tf.set_random_seed(SEED)
    spdot = tf.sparse_tensor_dense_matmul

print("Using TF {}".format(tf.__version__))
dot = tf.matmul
np.random.seed(SEED)


class GCN(object):
    def __init__(self, An, X, sizes, **kwargs):
        """ 
        Parameters
        ----------
        An : scipy.sparse matrix
            normalized adjacency matrix
        X : scipy.sparse matrix 
            feature matrix
        sizes : list
            size in each layer
        """

        self.An = sp_matrix_to_sp_tensor(An)
        self.X = sp_matrix_to_sp_tensor(X)
        self.layer_sizes = sizes
        self.shape = An.shape

        self.lr = kwargs.get('lr', 1e-3)
        self.with_relu = kwargs.get('with_relu', True)
        self.with_bias = kwargs.get('with_bias', True)
        self.dropout = kwargs.get('dropout', 0.5)
        self.verbose = kwargs.get('verbose', False)

        init_weight = tf.initializers.glorot_normal()

        self.W1 = tf.Variable(init_weight(shape=(self.shape[1], self.layer_sizes[0])))
        self.W2 = tf.Variable(init_weight(shape=(self.layer_sizes)))

        self.var_list = [self.W1, self.W2]
        if self.with_bias:
            self.b1 = tf.Variable(tf.zeros(self.layer_sizes[0],))
            self.b2 = tf.Variable(tf.zeros(self.layer_sizes[1],))
            self.var_list.extend([self.b1, self.b2])

    def train(self, idx_train, labels_train, n_iters=200):

        train_losses = []
        # use adam to optimize
        opt = tf.optimizers.Adam(learning_rate=self.lr)
        for it in range(n_iters):
            tic = time()
            with tf.GradientTape() as tape:
                _loss = self.loss_fn(idx_train, np.eye(2)[labels_train])
            grad_list = tape.gradient(_loss, self.var_list)
            grads_and_vars = zip(grad_list, self.var_list)
            opt.apply_gradients(grads_and_vars)

            # evaluate on the training
            train_loss, train_acc = self.evaluate(idx_train, labels_train)
            # TODO: evaluate on the validation
            # self.evaluate(idx_train, labels_train)

            # TODO: early stopping in the training process
            train_losses.append(train_loss)
            toc = time()
            if self.verbose:
                print("iter:{:03d}".format(it),
                      "train_loss:{:.4f}".format(train_loss),
                      "train_acc:{:.4f}".format(train_acc),
                      "time:{:.4f}".format(toc - tic))
        return train_losses

    def loss_fn(self, idx, labels):
        # TODO: build the layer form keras layer api
        # first layer
        _h1 = spdot(self.X, self.W1)
        _h1 = spdot(self.An, _h1)
        if self.with_bias:
            _h1 = tf.nn.bias_add(_h1, self.b1)
        self.h1 = tf.nn.relu(_h1)
        self.h1 = tf.nn.dropout(self.h1, self.dropout)

        # second layer
        _h2 = dot(self.h1, self.W2)
        _h2 = spdot(self.An, _h2)
        if self.with_bias:
            _h2 = tf.nn.bias_add(_h2, self.b2)
        self.h2 = _h2

        """ calculate the loss base on idx and labels """
        _logits = tf.gather(self.h2, idx)
        _loss_per_node = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                 logits=_logits)
        _loss = tf.reduce_mean(_loss_per_node)
        return _loss

    def evaluate(self, idx, true_labels):
        _loss = self.loss_fn(idx, np.eye(2)[true_labels]).numpy()
        _pred_logits = tf.gather(self.h2, idx)
        _pred_labels = tf.argmax(_pred_logits, axis=1).numpy()
        _acc = accuracy_score(_pred_labels, true_labels)
        return _loss, _acc


if __name__ == "__main__":
    import networkx as nx

    def _norm(A):
        A = A + sp.eye(A.shape[0])
        deg = A.sum(1).A1
        deg_inv = np.power(deg, -0.5)
        D_inv = sp.diags(deg_inv)
        return D_inv @ A @ D_inv

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G)

    An = _norm(A)
    X = sp.diags(A.sum(1).A1).tocsr()

    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    with tf.device("/device:GPU:1"):
        gcn = GCN(An, X, [16, 2], dropout=0, verbose=True)
        gcn.train([0, 33], [0, 1])
        test_res = gcn.evaluate(range(1, 33), y_true[1:33])
        print("Test loss {:.4f}".format(test_res[0]),
              "test acc {:.4f}".format(test_res[1]))

    # explore the dropout in the GCN
    # dr_losses = []
    # test_acc = []
    # for dr in np.arange(0, 1, 0.1):
    #     gcn = GCN(An, X, [16, 2], dropout=0.1, verbose=True)
    #     losses = gcn.train([0, 33], [0, 1])
    #     print(gcn.evaluate(range(1, 33), y_true[1:33]))

    #     dr_losses.append(losses)
    #     test_acc.append(gcn.evaluate(range(1, 33), y_true[1:33]))
    # import pickle
    # pickle.dump([dr_losses, test_acc], open('tmp.pkl', 'wb'))
