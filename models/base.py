from absl import flags
from models.layers import GraphConv
from models.utils import sp_matrix_to_sp_tensor
from sklearn.metrics import accuracy_score
from time import time
import numpy as np
import os
import scipy.sparse as sp
import tensorflow as tf
import warnings

SEED = 15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

tf.random.set_seed(SEED)
spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul

FLAGS = flags.FLAGS

# TODO: build the base class 
class Base(object):
    def __init__(self, **kwargs):
        self.with_relu = kwargs.get('with_relu', True)
        self.with_bias = kwargs.get('with_bias', True)
        
        self.lr = FLAGS.learning_rate
        self.dropout = FLAGS.dropout
        self.verbose = FLAGS.verbose

    def __call__(self, input):
        pass

    def _logger(self):
        pass
