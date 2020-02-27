from models.layers import GCNConv
from models.base import Base
from models.utils import split_edge, sp_matrix_to_sp_tensor, sparse_dropout, preprocess_graph
import tensorflow as tf
from absl import flags
from time import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

FLAGS = flags.FLAGS


class GAE(Base):
    def __init__(self, An, X, sizes, **kwargs):
        """
        """
        super(GAE, self).__init__(**kwargs)
        self.An = An
        self.X = X
        self.layer_size = sizes
        self.An_tf = sp_matrix_to_sp_tensor(self.An)
        self.X_tf = sp_matrix_to_sp_tensor(self.X)
        self.layer1 = GCNConv(self.layer_size[0], activation='relu')
        self.layer2 = GCNConv(self.layer_size[1])
        self.opt = tf.optimizers.Adam(learning_rate=self.lr)

    def train(self, A):
        """
        """
        # TODO: remove to preprocess file
        data = split_edge(A)
        self.A_train = data[0]
        # self.An_tf = sp_matrix_to_sp_tensor(preprocess_graph(self.A_train))
        train_pos_edges, train_neg_edges = data[1:3]
        val_pos_edges, val_neg_edges = data[3:5]
        test_pos_edges, test_neg_edges = data[5:]

        pos_weight = float(A.shape[0] ** 2 - A.sum()) / A.sum()
        norm = A.shape[0] ** 2 / float((A.shape[0] ** 2 - A.sum()) * 2)
        for epoch in range(FLAGS.epochs):
            tic = time()
            with tf.GradientTape() as tape:
                _loss = self.loss_fn(A, pos_weight=pos_weight, norm=norm)

            grad_list = tape.gradient(_loss, self.var_list)
            grad_and_vars = zip(grad_list, self.var_list)
            self.opt.apply_gradients(grad_and_vars)

            train_roc, train_ap = self.evaluate(train_pos_edges, train_neg_edges, A)
            val_roc, val_ap = self.evaluate(val_pos_edges, val_neg_edges, A)

            toc = time()
            if self.verbose:
                print("loss {:.4f}".format(_loss),
                      "roc {:.4f}".format(train_roc),
                      "ap {:.4f}".format(train_ap),
                      "roc {:.4f}".format(val_roc),
                      "ap {:.4f}".format(val_ap),
                      "time:{:.4f}".format(toc - tic))
        test_roc, test_ap = self.evaluate(test_pos_edges, test_neg_edges, A)
        print("roc {:.4f}".format(test_roc),
              "ap {:.4f}".format(test_ap))

    def loss_fn(self, A, pos_weight=1, norm=1, training=True):
        """ 
        """
        _X = sparse_dropout(self.X_tf, self.dropout, [self.X.nnz]) if training else self.X_tf
        self.h1 = self.layer1([self.An_tf, _X])
        _h1 = tf.nn.dropout(self.h1, self.dropout) if training else self.h1
        self.h2 = self.layer2([self.An_tf, _h1])
        self.z_mean = self.h2
        _emb = tf.nn.dropout(self.h2, self.dropout) if training else self.h2
        self.var_list = self.layer1.weights + self.layer2.weights
        self.P = tf.matmul(_emb, tf.transpose(_emb))
        _A = tf.constant(A.todense(), dtype=tf.float32)
        _loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(_A, self.P, pos_weight))

        return _loss

    def evaluate(self, pos_edge, neg_edge, A_orig):
        """  """
        P = tf.matmul(self.z_mean, tf.transpose(self.z_mean))
        pred_P = tf.sigmoid(P).numpy()
        pred_pos = pred_P[pos_edge[:, 0], pos_edge[:, 1]]
        pred_neg = pred_P[neg_edge[:, 0], neg_edge[:, 1]]

        preds_all = np.hstack([pred_pos, pred_neg])
        labels_all = np.hstack([np.ones(pos_edge.shape[0]), np.zeros(neg_edge.shape[0])])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score


class VGAE(Base):
    def __init__(self, An, X, sizes, **kwargs):
        """
        """
        super(VGAE, self).__init__(**kwargs)
        self.An = An
        self.X = X
        self.layer_size = sizes
        self.An_tf = sp_matrix_to_sp_tensor(self.An)
        self.X_tf = sp_matrix_to_sp_tensor(self.X)
        self.layer1 = GCNConv(self.layer_size[0], activation='relu')
        self.layer2 = GCNConv(self.layer_size[1])
        self.variational_layer = GCNConv(self.layer_size[1])
        self.opt = tf.optimizers.Adam(learning_rate=self.lr)

    def train(self, A):
        """
        """
        # TODO: remove to preprocess file
        data = split_edge(A)
        self.A_train = data[0]
        # self.An_tf = sp_matrix_to_sp_tensor(preprocess_graph(self.A_train))
        train_pos_edges, train_neg_edges = data[1:3]
        val_pos_edges, val_neg_edges = data[3:5]
        test_pos_edges, test_neg_edges = data[5:]

        pos_weight = float(A.shape[0] ** 2 - A.sum()) / A.sum()
        norm = A.shape[0] ** 2 / float((A.shape[0] ** 2 - A.sum()) * 2)
        for epoch in range(FLAGS.epochs):
            tic = time()
            with tf.GradientTape() as tape:
                _loss = self.loss_fn(A, pos_weight=pos_weight, norm=norm)

            grad_list = tape.gradient(_loss, self.var_list)
            grad_and_vars = zip(grad_list, self.var_list)
            self.opt.apply_gradients(grad_and_vars)

            train_roc, train_ap = self.evaluate(train_pos_edges, train_neg_edges, A)
            val_roc, val_ap = self.evaluate(val_pos_edges, val_neg_edges, A)

            toc = time()
            if self.verbose:
                print("loss {:.4f}".format(_loss),
                      "roc {:.4f}".format(train_roc),
                      "ap {:.4f}".format(train_ap),
                      "roc {:.4f}".format(val_roc),
                      "ap {:.4f}".format(val_ap),
                      "time:{:.4f}".format(toc - tic))
        test_roc, test_ap = self.evaluate(test_pos_edges, test_neg_edges, A)
        print("roc {:.4f}".format(test_roc),
              "ap {:.4f}".format(test_ap))

    def loss_fn(self, A, pos_weight=1, norm=1, training=True):
        """ 
        """
        _X = sparse_dropout(self.X_tf, self.dropout, [self.X.nnz]) if training else self.X_tf
        self.h1 = self.layer1([self.An_tf, _X])
        _h1 = tf.nn.dropout(self.h1, self.dropout) if training else self.h1
        self.h2 = self.layer2([self.An_tf, _h1])
        self.z_mean = self.h2
        # variational perturbation
        self.z_std = self.layer2([self.An_tf, _h1])
        self.z_std = tf.nn.dropout(self.z_std, self.dropout) if training else self.z_std
        self.z = self.z_mean + tf.random.normal([A.shape[0], self.layer_size[1]]) * tf.exp(self.z_std)

        _emb = tf.nn.dropout(self.z, self.dropout) if training else self.z
        self.var_list = self.layer1.weights + self.layer2.weights
        self.P = tf.matmul(_emb, tf.transpose(_emb))

        _A = tf.constant(A.todense(), dtype=tf.float32)
        _loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(_A, self.P, pos_weight))

        return _loss

    def evaluate(self, pos_edge, neg_edge, A_orig):
        """  """
        P = tf.matmul(self.z_mean, tf.transpose(self.z_mean))
        pred_P = tf.sigmoid(P).numpy()
        pred_pos = pred_P[pos_edge[:, 0], pos_edge[:, 1]]
        pred_neg = pred_P[neg_edge[:, 0], neg_edge[:, 1]]

        preds_all = np.hstack([pred_pos, pred_neg])
        labels_all = np.hstack([np.ones(pos_edge.shape[0]), np.zeros(neg_edge.shape[0])])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score
