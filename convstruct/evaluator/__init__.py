from convstruct.api.ops import adamOp
from os import path
from PIL import Image, ImageEnhance
import random
import os
import tensorflow as tf
import numpy as np
import shutil
import time
import math
from collections import deque


class Evaluator:
    def __init__(self, dims, limit, gpus, memory, location, name):
        """
        :param dims: list, output dimensions [batch_size, width, height, channels], only if image output.
        :param limit: int, maximum time limit per episode.
        :param gpus: int, amount of gpus to use.
        :param memory: int, amount of gpu memory to use, in gigabytes.
        :param location: str, directory name.
        :param name: str, string matching that of the name used for the created topology.
        """
        self.gpus = gpus
        self.memory = memory
        self.dims = dims
        self.limit = limit
        self.location = location
        self.name = name
        self.ckpt = self.getCheckPoint()
        self.current_limit = self.ckpt['limit']
        self.updates = self.ckpt['updates']
        self.epsilon = self.ckpt['epsilon']
        self.replay: list = self.ckpt['replay']
        self.repeats = 5
        self.bias_init = tf.constant_initializer(0.0)
        self.kernel_init = tf.initializers.variance_scaling(dtype=tf.float32)
        self.size = (((memory * 5) + 5) * 2) + 1
        self.lr = 0.0001

    def getCheckPoint(self):
        """
        :return: This function returns a created/loaded checkpoint dictionary.
        """
        if not os.path.exists(os.path.join(self.location, self.name)):
            os.makedirs(os.path.join(self.location, self.name))
        if not os.path.exists(os.path.join(os.path.join(self.location, self.name), 'temp')):
            os.makedirs(os.path.join(os.path.join(self.location, self.name), 'temp'))
        if not os.path.exists(os.path.join(os.path.join(self.location, self.name), 'evaluator')):
            os.makedirs(os.path.join(os.path.join(self.location, self.name), 'evaluator'))
        if path.exists(os.path.join(os.path.join(self.location, self.name), 'ckpt.npy')):
            ckpt = np.load(os.path.join(os.path.join(self.location, self.name), 'ckpt.npy')).flat[0]
            if ckpt['new']:
                ckpt['done'] = [False for _ in range(self.gpus)]
                ckpt['reset'] = [False for _ in range(self.gpus)]
        else:
            ckpt = {'replay': deque(maxlen=50000), 'updates': 0, 'epsilon': 1, 'limit': self.limit // 1000, 'episodes': 0, 'new': True, 'reset': [False for _ in range(self.gpus)], 'done': [False for _ in range(self.gpus)]}
            for gpu in range(self.gpus):
                ckpt['done_%d' % gpu] = False
        return ckpt

    def getSim(self, topology, energy=None, scores=None, loss=None, action=None, sims=None):
        """
        :param topology: dict, dictionary of for each topology that includes if it hit a nan value during training.
        :param energy: list of floats,
        :param scores: list of ints or floats, scores from evaluating model outputs, one for each gpu.
        :param loss: list of floats, latest returned loss from model, one for each gpu.
        :param action: list of ints, action of evaluator, chosen randomly or inferred, one for each gpu.
        :param sims: dict, dictionary containing all the graph elements of the current simulation.
        :return: This function returns a simulation dictionary.
        """
        if sims is not None:
            sims['parameter_ph'] = tf.placeholder(tf.float32, [None, self.size + 5], name='parameter_ph')
            sims['memory_ph'] = tf.placeholder(tf.float32, [None, 4], name='memory_ph')
            sims['main_ops'] = self.getRLGraph(sims['parameter_ph'], 'main')
            sims['target_ops'] = self.getRLGraph(sims['parameter_ph'], 'target')
            sims['train_ops'] = self.getRLLoss(sims['memory_ph'], sims['main_ops'], 'main')
            sims['graph'] = []
        for gpu in range(self.gpus):
            if self.ckpt['new']:
                self.ckpt['done_%d' % gpu] = False
                self.ckpt['save_model_%d' % gpu] = False
                self.ckpt['current_time_%d' % gpu] = 0
                self.ckpt['interval_time_%d' % gpu] = self.ckpt['limit'] // 10
                self.ckpt['reset'] = [False for _ in range(self.gpus)]
            self.ckpt['prev_ready_%d' % gpu] = False if self.ckpt['new'] or self.ckpt['reset'][gpu] else True
            self.ckpt['prev_loss_%d' % gpu] = None if self.ckpt['new'] or self.ckpt['reset'][gpu] else self.ckpt['prev_loss_%d' % gpu]
            self.ckpt['prev_acc_%d' % gpu] = None if self.ckpt['new'] or self.ckpt['reset'][gpu] else self.ckpt['prev_acc_%d' % gpu]
            self.ckpt['prev_weights_%d' % gpu] = [] if self.ckpt['new'] or self.ckpt['reset'][gpu] else self.ckpt['prev_weights_%d' % gpu]
            self.ckpt['prev_energy_%d' % gpu] = 0 if self.ckpt['new'] or self.ckpt['reset'][gpu] else self.ckpt['prev_energy_%d' % gpu]
            self.ckpt['total_time_%d' % gpu] = 0 if self.ckpt['new'] else (self.ckpt['total_time_%d' % gpu] + self.ckpt['current_time_%d' % gpu])
            self.ckpt['current_time_%d' % gpu] = 0
            self.ckpt['episode_duration_%d' % gpu] = 0 if self.ckpt['new'] or self.ckpt['reset'][gpu] else self.ckpt['episode_duration_%d' % gpu]
            self.ckpt['episode_reward_%d' % gpu] = 0 if self.ckpt['new'] else self.ckpt['episode_reward_%d' % gpu]
            if not self.ckpt['done_%d' % gpu] and scores is not None:
                self.ckpt['new_sim_%d' % gpu] = np.array(topology['%s_feed_%d' % (self.name, gpu)] + [energy[gpu], scores[gpu], loss[gpu], self.ckpt['episode_duration_%d' % gpu], self.ckpt['current_time_%d' % gpu]], dtype=np.float32).reshape(self.size + 5)
                self.replay.append((self.ckpt['current_sim_%d' % gpu], action[gpu], self.ckpt['episode_reward_%d' % gpu], self.ckpt['new_sim_%d' % gpu], self.ckpt['done'][gpu]))
                self.ckpt['done_%d' % gpu] = True if self.ckpt['done'][gpu] else False
            self.ckpt['current_sim_%d' % gpu] = np.array(topology['%s_feed_%d' % (self.name, gpu)] + [0, 0, 0, 0, 0], dtype=np.float32).reshape(self.size + 5) if self.ckpt['new'] or self.ckpt['reset'][gpu] else self.ckpt['new_sim_%d' % gpu]
        return sims

    def getSaver(self, sess, sims):
        """
        :param sess: tf.Session, the active episode session.
        :param sims: dict, dictionary containing all the graph elements of the current simulation.
        :return: This function returns the topology savers and restores previous weights conditionally.
        """
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        main_variables = [var for var in tf.global_variables() if var.name.startswith('main/')]
        main_saver = tf.train.Saver(main_variables)
        target_saver = tf.train.Saver(main_variables)
        if path.exists(os.path.join(self.location, os.path.join(self.name, 'rl.ckpt.meta'))):
            main_saver.restore(sess, os.path.join(self.location, os.path.join(self.name, 'rl.ckpt')))
            target_saver.restore(sess, os.path.join(self.location, os.path.join(self.name, 'rl.ckpt')))
        for gpu in range(self.gpus):
            if not self.ckpt['done'][gpu]:
                gpu_variables = [var for var in tf.global_variables() if var.name.startswith('%s/' % self.name if gpu == 0 else (self.name + '_' + str(gpu - 1)))]
                sims['saver_%d' % gpu] = tf.train.Saver(gpu_variables)
                if not self.ckpt['new'] and not self.ckpt['reset'][gpu]:
                    sims['saver_%d' % gpu].restore(sess, os.path.join(os.path.join(self.location, os.path.join(self.name, 'temp')), '%s.ckpt' % self.name if gpu == 0 else (self.name + '_' + str(gpu - 1))))
        self.ckpt['reset'] = [False for _ in range(self.gpus)]
        return sims, main_saver, target_saver

    def getAction(self, sess, sims):
        """
        :param sess: tf.Session, the active episode session.
        :param sims: dict, dictionary containing all the graph elements of the current simulation.
        :return: This function returns the a random action or the action the evaluator chose.
        """
        self.ckpt['new'] = self.decayEpsilon()
        self.ckpt['small'] = [False for _ in range(self.gpus)]
        self.ckpt['medium'] = [False for _ in range(self.gpus)]
        self.ckpt['large'] = [False for _ in range(self.gpus)]
        action: list = [0 for _ in range(self.gpus)]
        for gpu in range(self.gpus):
            if not self.ckpt['done'][gpu]:
                if np.random.random() > self.epsilon:
                    sess_feed = {sims['parameter_ph']: (np.array(self.ckpt['current_sim_%d' % gpu]).reshape([1, self.size + 5]))}
                    get_qs = sess.run(sims['main_ops'], sess_feed)
                    action[gpu] = np.argmax(get_qs)
                else:
                    action[gpu] = random.randint(0, 3 if np.random.random() > (self.epsilon + .2) else 2)
                actions = [(False if i != action[gpu] else True) for i in range(4)]
                self.ckpt['action'] = action[gpu]
                self.ckpt['episode_duration_%d' % gpu] += [10, 100, 200, 0][self.ckpt['action']]
                self.ckpt['small'][gpu], self.ckpt['medium'][gpu], self.ckpt['large'][gpu], self.ckpt['reset'][gpu] = actions[0], actions[1], actions[2], actions[3]
        return action

    def learnMemory(self, sess, sims, main_saver, target_saver):
        """
        :param sess: tf.Session, the active episode session.
        :param sims: dict, dictionary containing all the graph elements of the current simulation.
        :param main_saver: tf.train.Saver, the main graph active saver.
        :param target_saver: tf.train.Saver, the target graph active saver.
        :return: This function updates the reinforcement model graph used by the evaluator.
        """
        if len(self.replay) > 1000:
            mini_batch = random.sample(self.replay, 64)
            current_sim = np.array([transition[0] for transition in mini_batch])
            new_sim = np.array([transition[3] for transition in mini_batch])
            current_qs_list = sess.run(sims['main_ops'], {sims['parameter_ph']: current_sim})
            future_qs_list = sess.run(sims['target_ops'], {sims['parameter_ph']: new_sim})
            x, y = [], []
            for index, (current_sim, action, reward, new_sim, past_done) in enumerate(mini_batch):
                new_q = reward + ((0.99 * np.max(future_qs_list[index])) if not past_done else 0)
                current_qs = current_qs_list[index]
                current_qs[action] = new_q
                x.append(current_sim), y.append(current_qs)
            sess.run(sims['train_ops'], {sims['parameter_ph']: np.array(x), sims['memory_ph']: np.array(y)})
            update_check = 0
            for gpu in range(self.gpus):
                update_check += 1 if self.ckpt['done_%d' % gpu] else 0
            self.updates += 1 if update_check == self.gpus else 0
            if self.updates > 5:
                self.ckpt['limit'] += (self.limit // 1000) if self.limit > self.ckpt['limit'] else 0
                main_saver.save(sess, os.path.join(self.location, os.path.join(self.name, 'rl.ckpt')))
                target_saver.restore(sess, os.path.join(self.location, os.path.join(self.name, 'rl.ckpt')))
                self.updates = 0
        return

    def getReward(self, topology, scores, energy, losses):
        """
        :param topology: dict, dictionary of for each topology that includes if it hit a nan value during training.
        :param scores: list of ints or floats, scores from evaluating model outputs, one for each gpu.
        :param energy: list of floats, variation of weights from graph, one for each gpu.
        :param losses: list of floats, loss calculated from graph, one for each gpu.
        :return: This function returns the accumulating episode reward, one for each gpu.
        """
        for gpu in range(self.gpus):
            if not self.ckpt['done_%d' % gpu]:
                if (self.ckpt['total_time_%d' % gpu] + self.ckpt['current_time_%d' % gpu] >= self.ckpt['interval_time_%d' % gpu]) and self.ckpt['prev_ready_%d' % gpu]:
                    self.ckpt['episode_reward_%d' % gpu] += 3 if energy[gpu] < self.ckpt['prev_energy_%d' % gpu] and not self.ckpt['reset'][gpu] else -3
                    self.ckpt['episode_reward_%d' % gpu] += 4 if scores[gpu] > self.ckpt['prev_acc_%d' % gpu] and not self.ckpt['reset'][gpu] else -4
                    self.ckpt['episode_reward_%d' % gpu] += 3 if losses[gpu] < self.ckpt['prev_loss_%d' % gpu] and not self.ckpt['reset'][gpu] else -3
                    self.ckpt['interval_time_%d' % gpu] += (self.ckpt['limit'] // 10)
                    self.ckpt['prev_ready_%d' % gpu] = False
                if not self.ckpt['prev_ready_%d' % gpu]:
                    self.ckpt['prev_energy_%d' % gpu] = energy[gpu]
                    self.ckpt['prev_acc_%d' % gpu] = scores[gpu]
                    self.ckpt['prev_loss_%d' % gpu] = losses[gpu]
                    self.ckpt['prev_ready_%d' % gpu] = True
                if scores[gpu] >= 80:
                    self.ckpt['episode_reward_%d' % gpu] += (400 if scores[gpu] >= 90 else 300) if scores[gpu] >= 85 else 200
                    self.ckpt['done'][gpu] = True
                    self.ckpt['save_model_%d' % gpu] = True
                if self.ckpt['total_time_%d' % gpu] + self.ckpt['current_time_%d' % gpu] >= self.ckpt['limit']:
                    self.ckpt['episode_reward_%d' % gpu] -= 200 if self.ckpt['limit'] == self.limit else (((self.ckpt['interval_time_%d' % gpu] // (self.ckpt['limit'] // 10)) - 1) * 10)
                    self.ckpt['done'][gpu] = True
                if math.isnan(losses[gpu]):
                    self.ckpt['episode_reward_%d' % gpu] -= 400
                    self.ckpt['done'][gpu] = True
                print("Episode: %d | GPU: %d | " % (self.ckpt['episodes'] + gpu, gpu) + ("Time Out" if self.ckpt['total_time_%d' % gpu] + self.ckpt['current_time_%d' % gpu] >= self.ckpt['limit'] else ("NaN Found" if math.isnan(losses[gpu]) else ["Small Epoch", "Medium Epoch", "Large Epoch", "Reset"][self.ckpt['action']])) + " | Accuracy: %d | Loss: %f | Energy: %f | Reward: %d | Time remaining: %fs/%fs | Episode duration: %d" % (scores[gpu], losses[gpu], energy[gpu], self.ckpt['episode_reward_%d' % gpu], self.ckpt['total_time_%d' % gpu] + self.ckpt['current_time_%d' % gpu], self.ckpt['limit'], self.ckpt['episode_duration_%d' % gpu]))
                if self.ckpt['done'][gpu]:
                    for idx in range(len(topology['names'])):
                        os.rename(os.path.join(self.location, os.path.join(self.name, ('%s_topology_%d.jpg' % (topology['names'][idx], gpu)))), os.path.join(self.location, os.path.join(self.name, ('%s_topology_%d_reward_%d.jpg' % (topology['names'][idx], self.ckpt['episodes'] + gpu, self.ckpt['episode_reward_%d' % gpu])))))
        return self.ckpt

    def getRLGraph(self, parameters, name):
        """
        :param parameters: tf.placeholder, feed for graph parameters.
        :param name: str, name of graph.
        :return: This function returns the logit output from the estimate graph.
        """
        with tf.variable_scope(name):
            weight_1 = tf.Variable(tf.random_normal([self.size + 5, 128], stddev=0.03), name="weight_1")
            weight_2 = tf.Variable(tf.random_normal([128, 32], stddev=0.03), name="weight_2")
            weight_3 = tf.Variable(tf.random_normal([32, 8], stddev=0.03), name="weight_3")
            bias_1 = tf.Variable(tf.zeros([128]), name="bias_1")
            bias_2 = tf.Variable(tf.zeros([32]), name="bias_2")
            bias_3 = tf.Variable(tf.zeros([8]), name="bias_3")
            layer_1 = tf.nn.relu(tf.add(tf.matmul(parameters, weight_1), bias_1))
            layer_1 = tf.nn.dropout(layer_1, rate=0.5)
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weight_2), bias_2))
            layer_2 = tf.nn.dropout(layer_2, rate=0.5)
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weight_3), bias_3))
            logit = tf.layers.dense(inputs=layer_3, units=4)
        return logit

    def getRLLoss(self, memory, parameters, name):
        """
        :param memory: float32, replay action collected from each episode.
        :param parameters: float32, inferred action.
        :param name: str, name of graph.
        :return: This function returns the loss and training ops for the estimate model.
        """
        loss = tf.reduce_mean((memory - parameters) ** 2)
        train = adamOp(loss, name=name, learning_rate=self.lr)
        return train

    def decayEpsilon(self):
        """
        :return: This function returns a reduced epsilon value and a false boolean.
        """
        if self.epsilon > 0.001 and self.ckpt['new']:
            self.epsilon *= 0.99975
            self.epsilon = max(0.001, self.epsilon)
        return False

    def saveSession(self, sess, sims):
        """
        :param sess: tf.Session, the active episode session.
        :param sims: dict, dictionary containing all the graph elements of the current simulation.
        :return: This function returns the saved topology weights to a temp folder, one for each gpu.
        """
        for gpu in range(self.gpus):
            sims['saver_%d' % gpu].save(sess, os.path.join(os.path.join(self.location, os.path.join(self.name, 'temp')), '%s.ckpt' % self.name if gpu == 0 else (self.name + '_' + str(gpu - 1))))
        self.ckpt['updates'] = self.updates
        self.ckpt['epsilon'] = self.epsilon
        self.ckpt['replay'] = self.replay
        np.save(os.path.join(self.location, os.path.join(self.name, 'ckpt.npy')), self.ckpt)
        return

    def endEpisode(self, sess, topology):
        """
        :param sess: tf.session(), tensorflow active session.
        :param topology: dict, dictionary of for each topology that includes if it hit a nan value during training.
        :return: Returns a saved ckpt file, and a frozen .pb graph or deletes the temp folder.
        """
        if all(self.ckpt['done']):
            try:
                for gpu in range(self.gpus):
                    if self.ckpt['save_model_%d' % gpu]:
                        activation = topology['%s_output_node_%d' % (self.name, gpu)]
                        output_node_names = ['%s/%s' % (self.name if gpu == 0 else (self.name + '_' + str(gpu - 1)), activation if topology['%s_classes' % self.name] == 0 else 'logits/BiasAdd')]
                        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
                        with open(os.path.join(self.location, os.path.join(self.name, '%s_%d.pb' % (self.name, self.ckpt['episodes'] + gpu))), 'wb') as f:
                            f.write(frozen_graph_def.SerializeToString())
            except ValueError:
                print("The graph cannot be frozen as a .pb file since a tensor larger than 2gb has been found.")
            shutil.rmtree(os.path.join(self.location, os.path.join(self.name, 'temp')))
            os.makedirs(os.path.join(self.location, os.path.join(self.name, 'temp')))
            self.ckpt['episodes'] += 1 * self.gpus
            self.ckpt['new'] = True
            np.save(os.path.join(self.location, os.path.join(self.name, 'ckpt.npy')), self.ckpt)
        return

    def getScore(self, sess, evaluate, sims, start, topology, batch=None):
        """
        :param sess: tf.session(), tensorflow active session.
        :param evaluate: dict, dictionary of lists of outputs to score {'(gpu index)': [[real data 0, output 0] ... [real data X, output X]]}.
        :param sims: dict, dictionary containing all the graph elements of the current simulation.
        :param start: int, time.time() started at the beginning of the training loop.
        :param topology: dict, dictionary of for each topology that includes if it hit a nan value during training.
        :param batch: tf.train.batch, a batch of real data.
        :return: This function returns an evaluation of the data provided, using the evaluator model.
        """
        training = time.time()
        if not sims['graph'] and batch is not None:
            sims['phs'], sims['graph'] = self.initScore()
        if not os.path.exists(os.path.join(os.path.join(self.location, os.path.join(self.name, 'evaluator')), 'score.ckpt.meta')) and batch is not None:
            self.trainScore(sess, sims['phs'], sims['graph'], batch)
        elif topology['%s_classes' % self.name] == 0:
            self.loadScore(sess)
        omit = time.time() - training
        scores: list = [0 for _ in range(self.gpus)]
        energy: list = [0 for _ in range(self.gpus)]
        for gpu in range(self.gpus):
            if not self.ckpt['done_%d' % gpu]:
                scores[gpu] = self.calculateScore(sess, evaluate['%d' % gpu], sims, topology)
                energy[gpu] = self.calculateEnergy(sess, topology, gpu)
                self.ckpt['current_time_%d' % gpu] += (time.time() - start) - omit
        return scores, energy, sims

    def calculateEnergy(self, sess, topology, gpu):
        """
        :param sess: tf.session(), tensorflow active session.
        :param topology: dict, dictionary of for each topology that includes if it hit a nan value during training.
        :param gpu: int, the gpu index.
        :return: This function returns the mean absolute value of the current and previous weights of each convolution in the topology.
        """
        energy, idx = 0, 0
        for vertical in range(len(topology['%s_split_%d' % (self.name, gpu)])):
            for horizontal in range(topology['%s_split_%d' % (self.name, gpu)][vertical]):
                with tf.variable_scope('%s/input_%d_%d' % (self.name, horizontal, vertical), reuse=True):
                    w = tf.get_variable("kernel")
                    current = w.eval(session=sess)
                    if len(self.ckpt['prev_weights_%d' % gpu]) == int(topology['%s_size_%d' % (self.name, gpu)]):
                        energy += abs(np.mean(abs(current) - self.ckpt['prev_weights_%d' % gpu][idx]))
                        self.ckpt['prev_weights_%d' % gpu][idx] = current
                        idx += 1
                    else:
                        self.ckpt['prev_weights_%d' % gpu].append(current)
        return energy

    def initScore(self):
        """
        :return: This function returns the score tf.placeholders, batch, and graph.
        """
        train_ph = tf.placeholder(tf.bool, name='train_ph')
        real_ph = tf.placeholder(tf.float32, self.dims, name='real_ph')
        aug_ph = tf.placeholder(tf.float32, self.dims, name='aug_ph')
        phs = [train_ph, real_ph, aug_ph]
        real_pair = self.getScoreGraph(real_ph, train_ph)
        aug_pair = self.getScoreGraph(aug_ph, train_ph)
        loss = self.getScoreLoss(real_pair, aug_pair, 20, 5, 'score')
        logits = [tf.argmax(self.getScoreLogits(real_pair, 20, 'true' + str(i + 1)), axis=1) for i in range(5)]
        ops = adamOp(loss, name='score', learning_rate=1e-4)
        graph = [ops, logits, loss]
        return phs, graph

    def trainScore(self, sess, phs, graph, batch):
        """
        :param sess: tf.Session, the score graph session.
        :param phs: list, a list of score tf.placeholders.
        :param graph: list, [score graph ops, score logits, score loss].
        :param batch: tf.batch, a batch of ground truth data.
        :return: This function returns the trained weights for the evaluator's score model.
        """
        get_next = initBatch(batch, sess)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        score_variables = [var for var in tf.global_variables() if var.name.startswith('score/')]
        score = tf.train.Saver(score_variables)
        loss, epoch = 0, 0
        print("Training scoring model...")
        while True:
            if loss > -2 or epoch < 1000:
                real_batch = sess.run(get_next)
                paired_batch = np.zeros((real_batch.shape[0], real_batch.shape[1], real_batch.shape[2], 3))
                for i in range(real_batch.shape[0]):
                    paired_batch[i, :, :, :] = createBatchPair(real_batch[i, :, :, :])
                _, loss = sess.run([graph[0], graph[2]], {phs[0]: True, phs[1]: (real_batch + 1.0 / 2.0), phs[2]: (paired_batch + 1.0 / 2.0)})
                epoch += 1
                if epoch % 100 == 0 or (-2 >= loss and epoch >= 1000):
                    print("Epoch: %d |" % epoch + " Score loss: " + str(loss))
            else:
                break
        print("Scoring model saved.")
        score.save(sess, os.path.join(os.path.join(self.location, os.path.join(self.name, 'evaluator')), 'score.ckpt'))
        return

    def loadScore(self, sess):
        """
        :param sess: tf.Session, the score graph session.
        :return: This function returns pre-trained weights loaded into the active score graph.
        """
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        score_variables = [var for var in tf.global_variables() if var.name.startswith('score/')]
        score = tf.train.Saver(score_variables)
        score.restore(sess, os.path.join(os.path.join(self.location, os.path.join(self.name, 'evaluator')), 'score.ckpt'))
        return

    def calculateScore(self, sess, evaluate, sims, topology):
        """
        :param sess: tf.Session, the active evaluator session.
        :param evaluate: list, pairs of real and output data to evaluate and compare.
        :param sims: list, a list of score tf.placeholders and [score graph ops, score logits, score loss].
        :param topology: dict, dictionary of topology parameters to be fed to a graph.
        :return: This function returns a score based on the output provided, using the score model.
        """
        output_score = 0
        for i in range(len(evaluate)):
            eval_length = len(evaluate[i][0]) if topology['%s_classes' % self.name] > 0 else 5
            for ii in range(eval_length):
                if topology['%s_classes' % self.name] > 0:
                    output_score += 100 if np.argmax(evaluate[i][0][ii]) == np.argmax(evaluate[i][1][ii]) else 0
                else:
                    real_logits = sess.run(sims['graph'][1], {sims['phs'][0]: False, sims['phs'][1]: (evaluate[i][0] + 1.0 / 2.0)})
                    output_logits = sess.run(sims['graph'][1], {sims['phs'][0]: False, sims['phs'][1]: ((evaluate[i][1] * 255) + 1.0 / 2.0)})
                    noise_logits = sess.run(sims['graph'][1], {sims['phs'][0]: False, sims['phs'][1]: (np.random.rand(int(evaluate[i][1].shape[0]), self.dims[1], self.dims[1], int(evaluate[i][1].shape[3])).astype(np.float32) * 255) + 1.0 / 2.0})
                    real_set = set(sorted(real_logits[ii]))
                    output_set = set(sorted(output_logits[ii]))
                    noise_set = set(sorted(noise_logits[ii]))
                    output_score += 40 if len(real_set & output_set) > 0 else 0
                    output_score -= 10 if len(noise_set & output_set) > 0 and len(noise_set & real_set) == 0 else 0
            output_score /= eval_length
        output_score = (0 if 0 >= output_score else output_score) if 100 >= output_score else 100
        return output_score

    def getScoreGraph(self, data, train_ph):
        """
        :param data: tf.placeholder, input data to score.
        :param train_ph: tf.placeholder, tf.bool, controls batch normalization.
        :return: This function returns the score graph.
        """
        def layer(inputs, filters, training):
            conv2d = tf.layers.conv2d(inputs, filters=filters, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
            normalized = tf.layers.batch_normalization(conv2d, training=training)
            max_pool = tf.layers.max_pooling2d(normalized, pool_size=2, strides=2, padding="same")
            return max_pool

        with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
            data = tf.image.resize(data, [self.dims[1], self.dims[1]])
            data.set_shape([None, self.dims[1], self.dims[1], self.dims[3]])
            layer_1 = layer(data, 64, train_ph)
            layer_2 = layer(layer_1, 128, train_ph)
            layer_3 = layer(layer_2, 256, train_ph)
            layer_4 = layer(layer_3, 512, train_ph)
            out = tf.layers.flatten(layer_4)
            return out

    def getScoreLogits(self, data, num_clusters, identity):
        """
        :param data: tf.placeholder, input data to score.
        :param num_clusters: int, the number of clusters to average the scores from.
        :param identity: str, identity of the current cluster.
        :return: This function returns the score logits.
        """
        with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(inputs=data, units=num_clusters, activation=tf.nn.softmax, use_bias=True, kernel_initializer=self.kernel_init, bias_initializer=self.bias_init, name=(identity + 'dense'))
        return logits

    def getScoreLoss(self, real_pair, aug_pair, num_clusters, num_heads, head):
        """
        :param real_pair: tf.placeholder, the real data.
        :param aug_pair: tf.placeholder, an augmentation of the real data.
        :param num_clusters: int, the number of clusters to classify images into.
        :param num_heads: int, the number of clustering attempts.
        :param head: str, identity of the current cluster.
        :return: This function returns the loss for the score model.
        """
        loss = tf.constant(0, dtype=tf.float32)
        for i in range(num_heads):
            real_logits = self.getScoreLogits(real_pair, num_clusters, identity=head + str(i + 1))
            aug_logits = self.getScoreLogits(aug_pair, num_clusters, identity=head + str(i + 1))
            real_shape = real_logits.shape.as_list()[1]
            combined_logits = tf.transpose(real_logits) @ aug_logits
            averaged_logits = (combined_logits + tf.transpose(combined_logits)) / 2
            head_logits = tf.clip_by_value(averaged_logits, clip_value_min=1e-6, clip_value_max=tf.float32.max)
            head_logits /= tf.reduce_sum(head_logits)
            head_logits_x = tf.broadcast_to(tf.reshape(tf.reduce_sum(head_logits, axis=0), (real_shape, 1)), (real_shape, real_shape))
            head_logits_y = tf.broadcast_to(tf.reshape(tf.reduce_sum(head_logits, axis=1), (1, real_shape)), (real_shape, real_shape))
            loss += -tf.reduce_sum(head_logits * (tf.math.log(head_logits) - tf.math.log(head_logits_x) - tf.math.log(head_logits_y)))
        loss /= num_heads
        return loss


def createBatchPair(image):
    """
    :param image: RGB img, image data to create a pair from.
    :return: This function returns a transformed pair from image data provided, by sampling, rotating and cropping.
    """
    output = np.copy(np.array(image))
    f_v, s_v, d_v, c_v = random.uniform(0, 1), random.uniform(0, 2), random.uniform(0.7, 1), random.uniform(0, 1)
    output = np.flip(output, 0) if f_v > 0.5 else np.flip(output, 1)
    img = Image.fromarray((output * 255).astype(np.uint8))
    sharpen = ImageEnhance.Sharpness(img)
    sharper = sharpen.enhance(s_v)
    coloring = ImageEnhance.Color(sharper)
    colored = coloring.enhance(c_v)
    darken = ImageEnhance.Brightness(colored)
    new_pair = darken.enhance(d_v)
    return new_pair


def createConfig():
    """
    :return: Returns tensorflow configuration.
    """
    tf_config = tf.ConfigProto(allow_soft_placement=False)
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.polling_inactive_delay_msecs = 50
    return tf_config


def initBatch(dataset, sess):
    """
    :param dataset: tensor, tensor batch using files provided.
    :param sess: tf.Session, the active tensorflow session.
    :return: This function returns the next batch of data from the dataset.
    """
    if dataset is None:
        return None
    sess.run(dataset.initializer)
    return dataset.get_next()
