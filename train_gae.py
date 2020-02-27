from absl import app
from absl import flags
from models.autoencoder import GAE, VGAE

from models.utils import preprocess_graph, load_data, load_data_planetoid, split_edge
import numpy as np
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

try:
    import tensorflow.compat.v2 as tf
except ImportError as e:
    print(e)

print("Using TF {}".format(tf.__version__))
SEED = 15
np.random.seed(SEED)
tf.random.set_seed(SEED)

# let hyperpaprameters to be accessible in multiple modules
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_bool('verbose', False, 'Toogle the verbose.')
flags.DEFINE_bool('logging', False, 'Toggle the logging.')
flags.DEFINE_integer('gpu_id', 0, 'Specify the GPU id')


def main(argv):
    # config the CPU/GPU in TF, assume only one GPU is in use.
    # For multi-gpu setting, please refer to
    #   https://www.tensorflow.org/guide/gpu#using_multiple_gpus

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0 or FLAGS.gpu_id is None:
        device_id = "/device:CPU:0"
    else:
        tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu_id], 'GPU')
        device_id = '/device:GPU:0'

    A_mat, X_mat, z_vec, train_idx, val_idx, test_idx = load_data(FLAGS.dataset)
    An_mat = preprocess_graph(A_mat)

    # N = A_mat.shape[0]
    K = z_vec.max() + 1

    with tf.device(device_id):
        gae = VGAE(An_mat, X_mat, [FLAGS.hidden1, K])
        gae.train(A_mat)
        # test_res = gae.evaluate(test_idx, z_vec[test_idx], training=False)
        # W1 = gae.layer1.weight.numpy()
        # W2 = gae.layer2.weight.numpy()
        # print(np.linalg.norm(W1), np.linalg.norm(W2))
        # gcn = GCN(An_mat_diag, X_mat_stack, [FLAGS.hidden1, K])
        # gcn.train(train_idx_recal, z_vec[train_idx], val_idx_recal, z_vec[val_idx])
        # test_res = gcn.evaluate(test_idx_recal, z_vec[test_idx], training=False)
        # print("Dataset {}".format(FLAGS.dataset),
        #       "Test loss {:.4f}".format(test_res[0]),
        #       "test acc {:.4f}".format(test_res[1]))


if __name__ == "__main__":
    app.run(main)
