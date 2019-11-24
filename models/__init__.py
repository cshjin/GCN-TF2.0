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
