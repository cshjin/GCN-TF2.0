import tensorflow as tf
from tensorflow.keras import activations, regularizers, constraints, initializers
spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul


class GraphConv(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

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
        # gsize = input_shape[0][0]  # graph size
        fdim = input_shape[1][1]  # feature dim

        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                          shape=(fdim, self.units),
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
