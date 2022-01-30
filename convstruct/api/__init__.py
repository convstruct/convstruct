import sys
import os
import time
import numpy as np
import tensorflow as tf
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
from convstruct.api.core import Core
from convstruct.evaluator import Evaluator, initBatch, createConfig


def createTopology(dims, location, name, classes=0, flat=False, gpu=0, memory=8, struct=None):
    """
    :param dims: list, [batch size, input width, output width, output channels, input channels].
    :param location: string, directory name.
    :param name: string, unique label for created topology.
    :param classes: int, the number of output logits required from the graph.
    :param flat: bool, controls flat vs not flat dimensions (flat / not flat example: [10, 1] / [10, 10]).
    :param gpu: int, the gpu index.
    :param memory: int, amount of gpu memory to use, in gigabytes.
    :param struct: dict, dictionary of topologies to be used to create a graph.
    :return: This function returns a dictionary of topology parameters to be used to create a graph.
    """
    struct: dict = {'topology_index': 0, 'location': location, 'name': name, 'names': []} if struct is None else struct
    struct['%s_classes' % name] = classes
    struct['%s_%d' % (name, gpu)] = Core(struct, gpu, memory, struct['location'], name)
    struct['%s_memory_%d' % (name, gpu)] = (memory * 1000000000) + 1
    while True:
        if struct['%s_memory_%d' % (name, gpu)] > (memory * 1000000000):
            struct = struct['%s_%d' % (name, gpu)].getTopology(dims[0], dims[1], dims[2], dims[3], dims[4], flat)
        else:
            break
    struct['topology_index'] += 1
    struct['names'].append(name)
    np.save(os.path.join(os.path.join(struct['location'], os.path.join(struct['name'], 'temp')), 'topology.npy'), struct)
    return struct


def createGraph(graph_input, struct, input_shape, train_ph, location, name, gpu=0, memory=8, graph=None, alt=None, real=None):
    """
    :param graph_input: tf.batch/array or str, batch of inputs to be fed to a graph or string of previous graph output name.
    :param struct: dict, dictionary of topologies to be used to create a graph.
    :param input_shape: list, shape of inputs to be fed to a graph using a 4d-tensor: [batch size, width, height, channels].
    :param train_ph: tf.placeholder, tf.bool, controls batch normalization.
    :param location: string, directory name.
    :param name: string, string matching the name of the topology in struct to use.
    :param gpu: int, the gpu index.
    :param memory: int, amount of gpu memory to use, in gigabytes.
    :param graph: dict, dictionary of previous created graphs to be added upon.
    :param alt: string, unique string for topology output.
    :param real: tf.batch/array or str, batch of real inputs to compare graph outputs to.
    :return: This function returns a dictionary of a created graph.
    """
    graph = ({'real': real} if real is not None else dict()) if graph is None else graph
    graph['%s_%d' % (name, gpu)] = Core(struct, gpu, memory, location, name)
    graph_input = graph['%s_output_%d' % (graph_input, gpu)] if isinstance(graph_input, str) else graph_input
    with tf.device("/gpu:%d" % gpu):
        graph['%s_output_%d' % (name if alt is None else alt, gpu)] = graph['%s_%d' % (name, gpu)].setGraph(
            graph_input, input_shape, train_ph, np.array(struct['%s_split_%d' % (name, gpu)]))
    return graph


def createOps(graph, loss_feed, loss, optimizer, name, gpu=0):
    """
    :param graph: dict, dictionary of a created graph.
    :param loss_feed: list/string, string of loss input/s.
    :param loss: cs.ops.function, a loss function found in ops.py.
    :param optimizer: cs.ops.function, an optimizer function found in ops.py.
    :param name: string, string matching that of the name used for the created topology.
    :param gpu: int, the gpu index.
    :return: This function returns a graph dictionary with the created ops.
    """
    with tf.device("/gpu:%d" % gpu):
        graph['%s_loss_%d' % (name, gpu)] = loss(loss_feed, graph, gpu)
        graph['%s_ops_%d' % (name, gpu)] = optimizer(graph['%s_loss_%d' % (name, gpu)], name if gpu == 0 else (name + '_' + str(gpu - 1)))
    return graph


def createEpisode(setup, epoch, location, name, dims=None, limit=86400, gpus=1, memory=8):
    """
    :param setup: function, the function that returns the placeholders, topologies, graphs and batches.
    :param epoch: function, the function that returns an epoch training loop of the topologies.
    :param location: string, directory name.
    :param name: string, string matching the name of the struct.
    :param dims: list, output dimensions [batch_size, width, height, channels], only if image output.
    :param limit: int, maximum time limit per episode.
    :param gpus: int, amount of gpus to use.
    :param memory: int, amount of gpu memory to use, in gigabytes.
    :return: This function returns a completed episode.
    """
    temp_path = os.path.join(os.path.join(location, os.path.join(name, 'temp')), 'topology.npy')
    topology = np.load(temp_path).flat[0] if os.path.exists(temp_path) else None
    evaluator = Evaluator(dims, limit, gpus, memory, location, name)
    sims, ckpt = dict(), evaluator.getCheckPoint()
    while True:
        if not all(ckpt['done']):
            tf.reset_default_graph()
            phs, topology, graph, batch = setup(memory, None if ckpt['new'] or any(ckpt['reset']) else topology)
            sims = evaluator.getSim(topology, sims=sims)
            with tf.Session(config=createConfig()) as sess:
                sims, main_saver, target_saver = evaluator.getSaver(sess, sims)
                get_next = initBatch(batch, sess)
                while True:
                    if not any(ckpt['reset']) and not all(ckpt['done']):
                        start = time.time()
                        action = evaluator.getAction(sess, sims)
                        evaluate, loss = epoch(action, sess, phs, graph, get_next)
                        score, energy, sims = evaluator.getScore(sess, evaluate, sims, start, topology, batch)
                        ckpt = evaluator.getReward(topology, score, energy, loss)
                        evaluator.getSim(topology, energy, score, loss, action)
                        evaluator.learnMemory(sess, sims, main_saver, target_saver)
                        evaluator.saveSession(sess, sims)
                    else:
                        break
                evaluator.endEpisode(sess, topology)
        else:
            break
    return
