import sys
import os
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
import convstruct as cs
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

# ----- To repress Tensorflow deprecation warnings ----- #
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def argsParser():
    """
    :return: This function returns arguments for wpgan.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', type=int, default=6, help='Amount of gpu memory to use, in gigabytes.')
    parser.add_argument('--location', type=str, default='wgan', help='Name of directory.')
    parser.add_argument('--name', type=str, default='generator', help='Name of model.')
    parser.add_argument('--train', action='store_false', help='Toggle to allow for training.')
    parser.add_argument('--episodes', type=int, default=10000, help='Amount of episodes to complete.')
    return parser.parse_args()


class WGAN:
    def __init__(self):
        self.args = argsParser()

    def setup(self, memory, topology):
        """
        :param memory int, amount of gpu memory to use, in gigabytes.
        :param topology dict, dictionary of saved topology parameters to be re-fed to a graph.
        :return: This function returns the sentiment model tf.placeholders, topologies, graph and None, for batch.
        """
        if topology is None:
            g_topology = cs.createTopology([4, 8, 128, 3, 512], location=self.args.location, name='generator', memory=memory)
            topology = cs.createTopology([4, 128, 1, 512, 3], location=self.args.location, name='discriminator', classes=1, memory=memory, struct=g_topology)
        input_ph = tf.placeholder(tf.float32, [None, 100], name='input_ph')
        real_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='real_ph')
        training_ph = tf.placeholder(tf.bool, name='training')
        phs = [input_ph, real_ph, training_ph]
        batch = cs.createBatch('train_images', [128, 128, 3], batch_size=4)
        input_ph.set_shape([None, 100])
        w = tf.get_variable('w', shape=[100, 8 * 8 * 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', shape=[8 * 8 * 512], dtype=tf.float32, initializer=tf.zeros_initializer())
        flat_conv = tf.add(tf.matmul(input_ph, w), b, name='flat_conv')
        conv = tf.reshape(flat_conv, shape=[-1, 8, 8, 512], name='conv')
        embedding = tf.layers.batch_normalization(conv)
        embedding = tf.nn.relu(embedding)
        graph = cs.createGraph(embedding, topology, [None, 8, 8, 512], training_ph, location=self.args.location, name='generator', memory=wgan.args.memory)
        graph = cs.createGraph(real_ph, topology, [None, 128, 128, 3], training_ph, location=self.args.location, name='discriminator', memory=wgan.args.memory, graph=graph, alt='real')
        graph = cs.createGraph('generator', topology, [None, 128, 128, 3], training_ph, location=self.args.location, name='discriminator', memory=wgan.args.memory, graph=graph, alt='fake')
        penalty_epsilon = tf.random_uniform(shape=[4, 1, 1, 1], minval=0., maxval=1.)
        graph['penalty_output_0'] = real_ph + penalty_epsilon * (graph['generator_output_0'] - real_ph)
        graph = cs.createGraph('penalty', topology, [None, 128, 128, 3], training_ph, location=self.args.location, name='discriminator', memory=wgan.args.memory, graph=graph, alt='sum')
        graph = cs.createOps(graph, ['fake', 'real', 'sum', 'penalty'], loss=cs.gpLoss, optimizer=cs.adamOp, name='discriminator')
        graph = cs.createOps(graph, ['fake', None, None, None], loss=cs.gpLoss, optimizer=cs.adamOp, name='generator')
        return phs, topology, graph, batch

    def epoch(self, action, sess, phs, graph, batch):
        """
        :param action: list, int representing an action, per gpu.
        :param sess: tf.Session, the active episode session.
        :param phs: list, list of tf.placeholders for imagenet.
        :param graph: dict, dictionary of gpu batches of created graphs.
        :param batch: tf.data.Dataset, batch of images to feed as an input.
        :return: This function returns an epoch training loop of the sentiment topologies.
        """
        loss, evaluate, sess_out_d = [0], {'0': []}, 0
        for _ in range(cs.maxDuration(action)):
            real_batch = sess.run(batch)
            for _ in range(2):
                input_batch = np.random.uniform(-1, 1, [4, 100])
                sess_feed = {phs[0]: input_batch, phs[1]: (real_batch / 127.5) - 1, phs[2]: True}
                sess_out_d = sess.run([graph['discriminator_ops_0'], graph['discriminator_loss_0']], sess_feed)
            input_batch = np.random.uniform(-1, 1, [4, 100])
            sess_feed = {phs[0]: input_batch, phs[1]: (real_batch / 127.5) - 1, phs[2]: True}
            sess_out_g = sess.run([graph['%s_ops_0' % self.args.name], graph['%s_loss_0' % self.args.name]], sess_feed)
            loss[0] = sess_out_d[1] + sess_out_g[1]
        for _ in range(100):
            input_batch = np.random.uniform(-1, 1, [4, 100])
            real_batch = sess.run(batch)
            sess_feed = {phs[0]: input_batch, phs[1]: (real_batch / 127.5) - 1, phs[2]: False}
            g_out = sess.run(graph['%s_output_0' % self.args.name], sess_feed)
            evaluate['0'].append([real_batch, g_out])
        Image.fromarray(real_batch[0].astype(np.uint8)).save('img_real.png')
        Image.fromarray((((g_out[0] + 1) / 2) * 255).astype(np.uint8)).save('img_fake.png')
        return evaluate, loss


if __name__ == "__main__":
    wgan = WGAN()
    if wgan.args.train:
        for _ in range(wgan.args.episodes):
            cs.createEpisode(wgan.setup, wgan.epoch, location=wgan.args.location, name=wgan.args.name, dims=[4, 128, 128, 3], limit=172800, memory=wgan.args.memory)

tf.logging.set_verbosity(old_v)
