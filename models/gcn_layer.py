import warnings
import os
import numpy as np
import scipy.sparse as sp
from utils import sp_matrix_to_sp_tensor
from sklearn.metrics import accuracy_score

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
    tf.random.set_random_seed(SEED)
    spdot = tf.sparse_tensor_dense_matmul

dot = tf.matmul
np.random.seed(SEED)


class GCNLayer(tf.keras.layers.Layer):

    def __init__(self, units=32, act=lambda x: x, **kwargs):

        self.units = units
        self.activation = tf.keras.layers.Activation(act)
        self.with_bias = kwargs.get('with_bias', True)
        self.dropout = kwargs.get('dropout', 0.)
        self.K = 2
        super(GCNLayer, self).__init__()

    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        init_weight = tf.initializers.glorot_normal()
        fsize = input_shape[1][1]

        self.weight = self.add_weight(name="weight",
                                      shape=(fsize, self.units),
                                      initializer="glorot_normal",
                                      trainable=True)
        if self.with_bias:
            self.bias = self.add_weight(name="bias",
                                        shape=(self.units, ),
                                        initializer='zeros',
                                        trainable=True)
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        self.An = inputs[0]
        self.X = inputs[1]

        if isinstance(self.X, tf.SparseTensor):
            h = spdot(self.X, self.weight)
        else:
            h = dot(self.X, self.weight)
        if self.with_bias:
            h = tf.nn.bias_add(h, self.bias)

        if self.dropout:
            tf.nn.dropout(h, self.dropout)

        return self.activation(h)


if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G)
    An = sp_matrix_to_sp_tensor(A.astype('float32'))
    X = sp_matrix_to_sp_tensor(sp.diags(A.sum(1).A1))
    s = GCNLayer()
    s2 = GCNLayer(units=2)
    h1 = s([An, X])
    h2 = s2([An, h1])
    print(s2.weights)
