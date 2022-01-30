import convstruct as cs
import tensorflow as tf
import numpy as np
import pickle
import argparse
import os

# ----- To repress Tensorflow deprecation warnings ----- #
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def argsParser():
    """
    :return: This function returns arguments for imagenet.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', type=int, default=6, help='Amount of gpu memory to use, in gigabytes.')
    parser.add_argument('--location', type=str, default='imagenet', help='Name of directory.')
    parser.add_argument('--name', type=str, default='cnn', help='Name of model.')
    parser.add_argument('--train', action='store_false', help='Toggle to allow for training.')
    parser.add_argument('--episodes', type=int, default=10000, help='Amount of episodes to complete.')
    return parser.parse_args()


class cifar10:
    def __init__(self):
        self.args = argsParser()

    def setup(self, memory, topology):
        """
        :param memory int, amount of gpu memory to use, in gigabytes.
        :param topology dict, dictionary of saved topology parameters to be re-fed to a graph.
        :return: This function returns the tf.placeholders, batch, topologies, and graph for imagenet.
        """
        if topology is None:
            topology = cs.createTopology([32, 32, 1, 10, 3], location=self.args.location, name=self.args.name, classes=10, memory=memory)
        real_ph = tf.placeholder(tf.float32, [None, 32, 32, 3], name='real_ph')
        label_ph = tf.placeholder(tf.int32, [None, 10], name='label_ph')
        training_ph = tf.placeholder(tf.bool, name='training')
        phs = [real_ph, label_ph, training_ph]
        graph = cs.createGraph(real_ph, topology, [None, 32, 32, 3], training_ph, location=self.args.location, name=self.args.name, real=label_ph)
        graph = cs.createOps(graph, self.args.name, loss=cs.softmaxLoss, optimizer=cs.moOp, name=self.args.name)
        return phs, topology, graph, None

    def epoch(self, action, sess, phs, graph, batch):
        """
        :param action: list, int representing an action, per gpu.
        :param sess: tf.Session, the active episode session.
        :param phs: list, list of tf.placeholders for imagenet.
        :param graph: dict, dictionary of gpu batches of created graphs.
        :param batch: tf.data.Dataset, batch of images to feed as an input.
        :return: This function returns an epoch training loop of the sentiment topologies.
        """
        loss, evaluate = [0], {'0': []}
        train_x, train_y = get_dataset("train")
        test_x, test_y = get_dataset("test")
        for i in range(cs.maxDuration(action)):
            train_x_batch = train_x[i * 32: (i+1) * 32]
            train_y_batch = train_y[i * 32: (i+1) * 32]
            train_x_batch = np.reshape(train_x_batch, (32, 32, 32, 3))
            sess_feed = {phs[0]: train_x_batch, phs[1]: train_y_batch, phs[2]: True}
            sess_out = sess.run([graph['%s_ops_0' % self.args.name], graph['%s_loss_0' % self.args.name]], sess_feed)
            loss[0] = sess_out[1]
        for i in range(100):
            test_x_batch = test_x[i * 32: (i+1) * 32]
            test_y_batch = test_y[i * 32: (i+1) * 32]
            test_x_batch = np.reshape(test_x_batch, (32, 32, 32, 3))
            sess_feed = {phs[0]: test_x_batch, phs[1]: test_y_batch, phs[2]: True}
            logit_out = sess.run(graph['cnn_output_0'], sess_feed)
            evaluate['0'].append([test_y_batch, logit_out])
        return evaluate, loss


def get_dataset(name="train"):
    x, y = None, None
    f = open('cifar_10/batches.meta', 'rb')
    f.close()
    if name is "train":
        for i in range(5):
            f = open('cifar_10/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()
            _X = datadict["data"]
            _Y = datadict['labels']
            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)
            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)
    elif name is "test":
        f = open('cifar_10/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
        x = datadict["data"]
        y = np.array(datadict['labels'])
        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32 * 32 * 3)
    return x, dense_to_one_hot(y)


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


if __name__ == "__main__":
    cifar10 = cifar10()
    if cifar10.args.train:
        for _ in range(cifar10.args.episodes):
            cs.createEpisode(cifar10.setup, cifar10.epoch, location=cifar10.args.location, name=cifar10.args.name, limit=86400, memory=cifar10.args.memory)

tf.logging.set_verbosity(old_v)
