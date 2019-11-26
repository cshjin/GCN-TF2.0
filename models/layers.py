import warnings
import os
import numpy as np
import scipy.sparse as sp
from models.utils import sp_matrix_to_sp_tensor
from sklearn.metrics import accuracy_score

SEED = 15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

try:
    import tensorflow.compat.v2 as tf
    from tensorflow.keras import activations, regularizers, constraints, initializers
    tf.random.set_seed(SEED)
    spdot = tf.sparse.sparse_dense_matmul
except ImportError:
    import tensorflow as tf
    from tensorflow.keras import activations, regularizers, constraints, initializers
    tf.random.set_random_seed(SEED)
    spdot = tf.sparse_tensor_dense_matmul

dot = tf.matmul
np.random.seed(SEED)


class GraphConv(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None, **kwargs):

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(GraphConv, self).__init__()

    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        fsize = input_shape[1][1]

        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                          shape=(fsize, self.units),
                                          initializer=self.kernel_initializer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(name="bias",
                                            shape=(self.units, ),
                                            initializer=self.bias_initializer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        super(GraphConv, self).build(input_shape)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        self.An = inputs[0]
        self.X = inputs[1]

        if isinstance(self.X, tf.SparseTensor):
            h = spdot(self.X, self.weight)
        else:
            h = dot(self.X, self.weight)
        output = spdot(self.An, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        return output


if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G)
    with tf.device("/device:GPU:0"):
        An = sp_matrix_to_sp_tensor(A.astype('float32'))
        X = sp_matrix_to_sp_tensor(sp.diags(A.sum(1).A1))
        s = GraphConv(units=32)
        s2 = GraphConv(units=2)
        h1 = s([An, X])
        h2 = s2([An, h1])
        print(h1.shape, h2.shape)
