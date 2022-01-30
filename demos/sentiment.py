import convstruct as cs
import tensorflow as tf
import numpy as np
import argparse
import os
from os import path
from tensorflow.contrib import learn

# ----- To repress Tensorflow deprecation warnings ----- #
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def argsParser():
    """
    :return: This function returns arguments for sentiment analysis.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', type=int, default=6, help='Amount of gpu memory to use, in gigabytes.')
    parser.add_argument('--location', type=str, default='sentiment', help='Name of directory.')
    parser.add_argument('--name', type=str, default='twitter', help='Name of model.')
    parser.add_argument('--train', action='store_false', help='Toggle to allow for training.')
    parser.add_argument('--episodes', type=int, default=10000, help='Amount of episodes to complete.')
    return parser.parse_args()


class Sentiment:
    def __init__(self):
        self.args = argsParser()

    def setup(self, memory, topology):
        """
        :param memory int, amount of gpu memory to use, in gigabytes.
        :param topology dict, dictionary of saved topology parameters to be re-fed to a graph.
        :return: This function returns the sentiment model tf.placeholders, topologies, graph and None, for batch.
        """
        if topology is None:
            topology = cs.createTopology([32, 119, 1, 2, 1], location=self.args.location, name=self.args.name, classes=2, flat=True, memory=memory)
        real_ph = tf.placeholder(tf.float32, [None, 119, 1, 1], name='real_ph')
        label_ph = tf.placeholder(tf.float32, [None, 2], name='label_ph')
        training_ph = tf.placeholder(tf.bool, name='training')
        phs = [real_ph, label_ph, training_ph]
        graph = cs.createGraph(real_ph, topology, [None, 119, 1, 1], training_ph, location=self.args.location, name=self.args.name, real=label_ph)
        graph['%s_output_0' % self.args.name] = tf.nn.dropout(graph['%s_output_0' % self.args.name], 0.5)
        cell = tf.contrib.rnn.LSTMCell(100, state_is_tuple=True)
        expanded_output = tf.expand_dims(graph['%s_output_0' % self.args.name], -1)
        val, state = tf.nn.dynamic_rnn(cell, expanded_output, dtype=tf.float32)
        val2 = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val2, int(val2.get_shape()[0]) - 1)
        out_weight = tf.Variable(tf.random_normal([100, 2]))
        out_bias = tf.Variable(tf.random_normal([2]))
        graph['%s_output_0' % self.args.name] = tf.nn.xw_plus_b(last, out_weight, out_bias, name="scores")
        graph = cs.createOps(graph, self.args.name, loss=cs.softmaxLoss, optimizer=cs.adamOp, name=self.args.name)
        return phs, topology, graph, None

    def epoch(self, action, sess, phs, graph, batch):
        """
        :param action: list, int representing an action, per gpu.
        :param sess: tf.Session, the active episode session.
        :param phs: list, list of tf.placeholders for imagenet.
        :param graph: dict, dictionary of gpu batches of created graphs.
        :param batch: N/A.
        :return: This function returns an epoch training loop of the sentiment topologies.
        """
        loss, evaluate = [0], {'0': []}
        x_train, y_train, x_val, y_val = prepare_data(self.args)
        train_batches = gen_batch(list(zip(x_train, y_train)), 32, cs.maxDuration(action))
        val_batches = gen_batch(list(zip(x_val, y_val)), 32, 100)
        for duration, batch in enumerate(train_batches):
            if duration == cs.maxDuration(action):
                break
            train_x_gen, train_y_gen = zip(*batch)
            train_x_gen = np.reshape(train_x_gen, [32, 119, 1, 1])
            sess_feed = {phs[0]: train_x_gen, phs[1]: train_y_gen, phs[2]: True}
            sess_out = sess.run([graph['%s_ops_0' % self.args.name], graph['%s_loss_0' % self.args.name]], sess_feed)
            loss[0] = sess_out[1]
        for duration, batch in enumerate(val_batches):
            if duration == (100 if cs.maxDuration(action) != 0 else 0):
                break
            x_gen, y_gen = zip(*batch)
            x_gen = np.reshape(x_gen, [32, 119, 1, 1])
            sess_feed = {phs[0]: x_gen, phs[1]: y_gen, phs[2]: False}
            graph_out = sess.run(graph['%s_output_0' % self.args.name], sess_feed)
            y_out = sess.run(tf.nn.softmax(graph_out))
            evaluate['0'].append([y_gen, y_out])
        return evaluate, loss


def prepare_data(args):
    x_text = np.load("comments.npy")
    y = np.load("labels.npy")
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    np.random.seed(42)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(.10 * float(len(y)))
    x_train, x_val = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_val = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    if not path.exists(os.path.join(os.path.join(args.location, os.path.join(args.name, 'temp')), 'vocab')):
        vocab_processor.save(os.path.join(os.path.join(args.location, os.path.join(args.name, 'temp')), 'vocab'))
    return x_train, y_train, x_val, y_val


def gen_batch(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for _ in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    sentiment = Sentiment()
    if sentiment.args.train:
        for _ in range(sentiment.args.episodes):
            cs.createEpisode(sentiment.setup, sentiment.epoch, location=sentiment.args.location, name=sentiment.args.name, limit=86400, memory=sentiment.args.memory)

tf.logging.set_verbosity(old_v)
