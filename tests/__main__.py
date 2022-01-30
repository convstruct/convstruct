import sys
import os
from os import path
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
import convstruct as cs
import tensorflow as tf
import numpy as np
from PIL import Image
import shutil
import unittest


# ----- To repress Tensorflow deprecation warnings ----- #
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Test(unittest.TestCase):
    def test_a_build_log_dir(self):
        if os.path.exists(os.path.join('tests', '8to64')):
            shutil.rmtree(os.path.join('tests', '8to64'))
        if os.path.exists(os.path.join('tests', 'test_data')):
            shutil.rmtree(os.path.join('tests', 'test_data'))
        os.makedirs(os.path.join('tests', '8to64'))
        os.makedirs(os.path.join('tests', 'test_data'))
        os.makedirs(os.path.join(os.path.join('tests', '8to64'), 'temp'))
        self.assertTrue(path.exists('tests/test_data'))
        return

    def test_b_create_batch(self):
        for i in range(32):
            random_values = np.random.rand(64, 64, 3) * 255
            Image.fromarray(random_values.astype(np.uint8)).save('tests/test_data/img_%d.png' % i)
        get_batch_random = cs.createBatch(os.path.join('tests', 'test_data'), [64, 64, 3], 4, shuffle=False)
        real_png = np.asarray(Image.open('tests/test_data/img_0.png'))
        with tf.Session() as sess:
            get_next = cs.initBatch(get_batch_random, sess)
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            img_png_out = sess.run(get_next)
            converted = img_png_out.astype(int)
        self.assertTrue((real_png == converted[0]).all())
        return

    def test_c_create_topology(self):
        topology = cs.createTopology([4, 8, 64, 3, 3], location='tests', name='8to64')
        assert "8to64_output_node_0" in topology
        return

    def test_d_create_graph(self):
        topology = np.load(os.path.join(os.path.join(os.path.join('tests', '8to64'), 'temp'), 'topology.npy')).flat[0]
        input_ph = tf.placeholder(tf.float32, [None, 8, 8, 3], name='input_ph')
        training_ph = tf.placeholder(tf.bool, name='training')
        real_ph = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_ph')
        graph = cs.createGraph(input_ph, topology, [None, 8, 8, 3], training_ph, location='tests', name='8to64', real=real_ph)
        assert "8to64_output_0" in graph
        return topology, input_ph, training_ph, graph

    def test_e_create_ops(self):
        _, _, _, graph = self.test_d_create_graph()
        graph = cs.createOps(graph, '8to64', loss=cs.mseLoss, optimizer=cs.adamOp, name='8to64')
        assert "8to64_ops_0" in graph
        return

    def test_f_clean_test(self):
        shutil.rmtree(os.path.join('tests', os.path.join('8to64', 'temp')))
        os.makedirs(os.path.join('tests', os.path.join('8to64', 'temp')))
        tf.reset_default_graph()
        return

    def test_f_create_episode(self):
        def setup(memory, topology):
            if topology is None:
                self.test_c_create_topology()
            topology, input_ph, training_ph, graph = self.test_d_create_graph()
            phs = [input_ph, training_ph]
            batch = cs.createBatch(os.path.join('tests', 'test_data'), [64, 64, 3], batch_size=4)
            return phs, topology, graph, batch

        def epoch(action, sess, phs, graph, batch):
            loss = [np.random.rand(1).astype(np.float32)]
            evaluate = {'0': []}
            for _ in range(100):
                input_batch = np.random.rand(4, 64, 64, 3).astype(np.float32) * 255
                y_out = np.random.rand(4, 64, 64, 3).astype(np.float32) * 255
                evaluate['0'].append([input_batch, y_out])
            return evaluate, loss

        for _ in range(2):
            cs.createEpisode(setup, epoch, location='tests', name='8to64', dims=[4, 64, 64, 3], limit=20000)
        self.assertTrue(path.exists('tests/8to64/evaluator/score.ckpt.meta'))
        return


if __name__ == '__main__':
    unittest.main()
