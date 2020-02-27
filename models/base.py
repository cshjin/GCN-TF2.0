from absl import flags
FLAGS = flags.FLAGS


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

    def evaluate(self):
        raise NotImplementedError

    def loss_fn(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
