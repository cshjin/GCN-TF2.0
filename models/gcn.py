from absl import flags
from models.base import Base
from models.layers import GCNConv
from models.utils import sp_matrix_to_sp_tensor, sparse_dropout
from sklearn.metrics import accuracy_score
from time import time
import numpy as np
import tensorflow as tf

spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul

FLAGS = flags.FLAGS
tf.get_logger().setLevel('ERROR')


class GCN(Base):
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
        super().__init__(**kwargs)
        self.An = An
        self.X = X
        self.layer_sizes = sizes
        self.shape = An.shape

        self.An_tf = sp_matrix_to_sp_tensor(self.An)
        self.X_tf = sp_matrix_to_sp_tensor(self.X)

        self.layer1 = GCNConv(self.layer_sizes[0], activation='relu')
        self.layer2 = GCNConv(self.layer_sizes[1])
        self.opt = tf.optimizers.Adam(learning_rate=self.lr)

    def train(self, idx_train, labels_train, idx_val, labels_val):
        """ Train the model
        idx_train : array like
        labels_train : array like
        """
        K = labels_train.max() + 1
        train_losses = []
        val_losses = []
        # use adam to optimize
        for it in range(FLAGS.epochs):
            tic = time()
            with tf.GradientTape() as tape:
                _loss = self.loss_fn(idx_train, np.eye(K)[labels_train])

            # optimize over weights
            grad_list = tape.gradient(_loss, self.var_list)
            grads_and_vars = zip(grad_list, self.var_list)
            self.opt.apply_gradients(grads_and_vars)

            # evaluate on the training
            train_loss, train_acc = self.evaluate(idx_train, labels_train, training=True)
            train_losses.append(train_loss)
            val_loss, val_acc = self.evaluate(idx_val, labels_val, training=False)
            val_losses.append(val_loss)
            toc = time()
            if self.verbose:
                print("iter:{:03d}".format(it),
                      "train_loss:{:.4f}".format(train_loss),
                      "train_acc:{:.4f}".format(train_acc),
                      "val_loss:{:.4f}".format(val_loss),
                      "val_acc:{:.4f}".format(val_acc),
                      "time:{:.4f}".format(toc - tic))
        return train_losses

    def loss_fn(self, idx, labels, training=True):
        """ Calculate the loss function

        Parameters
        ----------
        idx : array like
        labels : array like

        Returns
        -------
        _loss : scalar
        """
        if training:
            _X = sparse_dropout(self.X_tf, self.dropout, [self.X.nnz])
        else:
            _X = self.X_tf

        self.h1 = self.layer1([self.An_tf, _X])
        if training:
            _h1 = tf.nn.dropout(self.h1, self.dropout)
        else:
            _h1 = self.h1

        self.h2 = self.layer2([self.An_tf, _h1])
        self.var_list = self.layer1.weights + self.layer2.weights
        # calculate the loss base on idx and labels
        _logits = tf.gather(self.h2, idx)
        _loss_per_node = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                 logits=_logits)
        _loss = tf.reduce_mean(_loss_per_node)
        # the weight_dacay only applys to the first layer.
        #         Same as the original implementation of GCN.
        # _loss += FLAGS.weight_decay * sum(map(tf.nn.l2_loss, self.var_list))
        _loss += FLAGS.weight_decay * sum(map(tf.nn.l2_loss, self.layer1.weights))
        return _loss

    def evaluate(self, idx, true_labels, training):
        """ Evaluate the model

        Parameters
        ----------
        idx : array like
        true_labels : true labels

        Returns
        -------
        _loss : scalar
        _acc : scalar
        """
        K = true_labels.max() + 1
        _loss = self.loss_fn(idx, np.eye(K)[true_labels], training=training).numpy()
        _pred_logits = tf.gather(self.h2, idx)
        _pred_labels = tf.argmax(_pred_logits, axis=1).numpy()
        _acc = accuracy_score(_pred_labels, true_labels)
        return _loss, _acc
