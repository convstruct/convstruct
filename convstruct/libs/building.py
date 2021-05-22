import math
import imageio
from itertools import chain
from .util import *
from .graphing import Graphing


class Building:
    def __init__(self, args, location, iicc, specifications, growth):
        """
        :param args: dictionary of arguments.
        :param location: path of directory to save weights and summaries to.
        :param iicc: classifier trained using convstruct.learn().
        :param specifications: dictionary of convstruct settings and system details.
        :param growth: dictionary of trained modules to reuse.
        """
        self.args = args
        self.location = location
        self.iicc = iicc
        self.specifications = specifications
        self.growth = growth
        self.size = (self.specifications['max_module_size'] * 2) + 1

    def active_graph(self, ground, iicc_learning, struct, stage):
        """
        :param ground: ground truth images.
        :param iicc_learning: boolean for enabling or disabling batch normalization in the iicc classifier.
        :param struct: dictionary of topology parameters to be fed to generator and discriminator modules.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function returns dictionaries of gpu batches of module and model training flows.
        """
        start, graph, disc, gen, magic_output, gpus, learning = time.time(), dict(), dict(), dict(), dict(), [], True if stage != 4 else False
        graph['discriminator_model_learning'] = tf.placeholder(tf.bool, name='discriminator_model_learning')
        graph['generator_model_learning'] = tf.placeholder(tf.bool, name='generator_model_learning')
        graph['starting_points'] = multi_placeholders(self.args['num_in'], 'starting_points', self.args['indir'])
        epoch_feed = {iicc_learning: False, graph['discriminator_model_learning']: learning, graph['generator_model_learning']: learning}
        for gpu in range(self.specifications['gpus'] if stage == 2 else 1):
            graph['discriminator_%d' % gpu] = Graphing(self.args, gpu, 'discriminator_model_memory_%d' % gpu, graph['discriminator_model_learning'], struct['discriminator_factor_%d' % gpu], self.growth, np.array(struct['teacher_2_split_%d' % gpu]), stage)
            graph['generator_%d' % gpu] = Graphing(self.args, gpu, 'generator_model_memory_%d' % gpu, graph['generator_model_learning'], struct['generator_factor_%d' % gpu], self.growth, np.array(struct['teacher_1_split_%d' % gpu]), stage)
            with tf.device(("/gpu:%d" % gpu) if self.specifications['gpus'] >= (gpu + 1) and not self.specifications['multi_gpu_test'] else "/gpu:0"):
                magic_output['magic_output_%d' % gpu] = graph['generator_%d' % gpu].magic(graph['starting_points'], self.args['num_in'])
                graph['magic_real_%d' % gpu] = graph['discriminator_%d' % gpu].magic(ground, self.args['num_comp'], 1)
                graph['magic_fake_%d' % gpu] = graph['discriminator_%d' % gpu].magic(magic_output['magic_output_%d' % gpu], self.args['num_comp'], 1)
                if self.args['num_comp'] > 1:
                    grounded, generated = tf.concat(list(ground.values()), 3), tf.concat(magic_output['magic_output_%d' % gpu], 3)
                graph['sum_hat_%d' % gpu] = tf.random_uniform([], 0.0, 1.0) * (grounded if self.args['num_comp'] > 1 else ground) + (1 - tf.random_uniform([], 0.0, 1.0)) * (generated if self.args['num_comp'] > 1 else magic_output['magic_output_%d' % gpu])
                graph['sum_logits_%d' % gpu] = graph['discriminator_%d' % gpu].magic(graph['sum_hat_%d' % gpu], self.args['num_comp'], 1)
                graph['magic_loss_%d' % gpu] = graph['discriminator_%d' % gpu].magic_loss(graph['magic_fake_%d' % gpu], graph['magic_real_%d' % gpu], graph['sum_hat_%d' % gpu], graph['sum_logits_%d' % gpu], self.classify(iicc_learning, ground, stage), self.classify(iicc_learning, magic_output['magic_output_%d' % gpu], stage))
                graph['magic_output_loss_%d' % gpu] = graph['generator_%d' % gpu].magic_loss(graph['magic_fake_%d' % gpu])
                disc['discriminator_ops_%d' % gpu] = graph['discriminator_%d' % gpu].optimize(graph['magic_loss_%d' % gpu])
                gen['generator_ops_%d' % gpu] = graph['generator_%d' % gpu].optimize(graph['magic_output_loss_%d' % gpu])
        graph['gen'], graph['disc'] = gen, disc
        performance_tracker(start, 'Graph completed in %fs')
        return graph, magic_output, epoch_feed

    def classify(self, iicc_learning, content, stage):
        """
        :param iicc_learning: boolean for enabling or disabling batch normalization in the iicc classifier.
        :param content: the image content to be classified.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function returns the flattened early output from the small cluster of the content.
        """
        start = time.time()
        content = (tf.concat(list(content.values()), 3) if type(content) is dict else tf.concat(content, 3)) if self.args['num_comp'] > 1 else content
        content = 255.0 * (0.5 * (content + 1.0))
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3 * self.args['num_comp']], name='img_mean')
        processed_content = content - mean
        processed_content = processed_content[:, :, :, ::-1]
        small_cluster = self.iicc.small_cluster(processed_content, iicc_learning, self.growth['x_size'] if stage != 2 else self.growth['small_x_size'])
        performance_tracker(start, 'IICC graph completed in %fs')
        return small_cluster

    def living(self, teacher, stage, quality=None):
        """
        :param teacher: tensorflow placeholder for a graph feed.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :param quality: tensorflow placeholder for a quality score pair for the matching graph feed.
        :return: This function returns the graph ops for the module estimator.
        """
        weight_1 = tf.Variable(tf.random_normal([self.size, 128], stddev=0.03), name="weight_1")
        weight_2 = tf.Variable(tf.random_normal([128, 32], stddev=0.03), name="weight_2")
        weight_3 = tf.Variable(tf.random_normal([32, 8], stddev=0.03), name="weight_3")
        bias_1 = tf.Variable(tf.zeros([128]), name="bias_1")
        bias_2 = tf.Variable(tf.zeros([32]), name="bias_2")
        bias_3 = tf.Variable(tf.zeros([8]), name="bias_3")
        with tf.variable_scope('live'):
            layer_1 = tf.nn.relu(tf.add(tf.matmul(teacher, weight_1), bias_1))
            layer_1 = tf.nn.dropout(layer_1, rate=0.5)
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weight_2), bias_2))
            layer_2 = tf.nn.dropout(layer_2, rate=0.5)
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weight_3), bias_3))
            logit = tf.layers.dense(inputs=layer_3, units=1)
        quality_estimate = logit
        loss = tf.reduce_mean((quality - quality_estimate)**2) if stage == 2 else None
        train = tf.train.AdamOptimizer(0.0001).minimize(loss) if stage == 2 else None
        return train, loss, quality_estimate

    def module_estimator(self, teacher, module_type, live_ops, struct, gpu):
        """
        :param teacher: tensorflow placeholder for a graph feed.
        :param module_type: value indicator to identify between generator(1) and discriminator(2) modules.
        :param live_ops: runs the inference on a graph feed to provide a score.
        :param struct: dictionary of topology parameters to be fed to generator and discriminator modules.
        :param gpu: the gpu index.
        :return: This function returns the estimated quality of a modules output.
        """
        with tf.Session(config=tf_config_setup()) as sess:
            coord, threads = initialization(sess)
            start = time.time()
            variables_to_restore = [var for var in tf.global_variables() if var.name.startswith('weights_') or var.name.startswith('bias_')]
            live = tf.train.Saver(variables_to_restore)
            if path.exists(os.path.join(self.location, 'live.ckpt.meta')):
                live.restore(sess, os.path.join(self.location, 'live.ckpt'))
            teacher_batch = np.reshape(np.array([struct['teacher_feed_%d_%d' % (module_type, gpu)]]), (1, (self.specifications['max_module_size'] * 2) + 1))
            live_feed = {teacher: teacher_batch}
            quality_score = sess.run(live_ops, live_feed)
            quality_score = 100 if quality_score[0][0] * 100 > 100.0 else (quality_score[0][0] * 100 if quality_score[0][0] * 100 > 0 else 0)
            performance_tracker(start, 'Module estimator completed in %fs')
            coord.request_stop()
            coord.join(threads)
        return quality_score

    def template(self, module_type, struct, gpu):
        """
        :param module_type: value indicator to identify between generator(1) and discriminator(2) modules.
        :param struct: dictionary of topology parameters to be fed to generator and discriminator modules.
        :param gpu: the gpu index.
        :return: This function returns a latent module and conditionally saves it to an index.
        """
        total_split, total_parameters = [], []
        seed = random.randrange(1000)
        np.random.seed(seed)
        struct['teacher_%d_module_%d' % (module_type, gpu)] = np.random.uniform(3 * self.growth['scaling'], self.specifications['max_module_size'], [1])
        struct['teacher_%d_parameters_%d' % (module_type, gpu)] = np.reshape([random.uniform(0.1, 1.0) for _ in range(int(struct['teacher_%d_module_%d' % (module_type, gpu)]) + 2)], (1, int(struct['teacher_%d_module_%d' % (module_type, gpu)]) + 2))
        struct['teacher_%d_split_%d' % (module_type, gpu)] = split_random(int(struct['teacher_%d_module_%d' % (module_type, gpu)]) // 3, int(np.random.uniform(self.growth['scaling'], int(struct['teacher_%d_module_%d' % (module_type, gpu)]) // 3, [1])[0]))
        total_split.append((struct['teacher_%d_split_%d' % (module_type, gpu)] + [0] * self.specifications['max_module_size'])[:self.specifications['max_module_size']])
        total_parameters.append((list(chain.from_iterable(struct['teacher_%d_parameters_%d' % (module_type, gpu)].tolist())) + [0] * self.specifications['max_module_size'])[:self.specifications['max_module_size']])
        return total_split, total_parameters

    def completing(self, name, module_type, struct, gpu):
        """
        :param name: string to indicate a generator or discriminator module.
        :param module_type: value indicator to identify between generator(1) and discriminator(1) modules.
        :param struct: dictionary of topology parameters to be fed to generator and discriminator modules.
        :param gpu: the gpu index.
        :return: This function returns the total ops of the generator or discriminator model and adds the modules to a dictionary.
        """
        struct['%s_factor_%d' % (name, gpu)], struct['%s_ops_%d' % (name, gpu)] = self.complete(struct['teacher_%d_parameters_%d' % (module_type, gpu)], int(struct['teacher_%d_module_%d' % (module_type, gpu)]) + 2, [struct['teacher_%d_split_%d' % (module_type, gpu)]], gpu, module_type)
        return
    
    def complete(self, parameters, module_size, split, gpu, module_type):
        """
        :param parameters: generated module topology array.
        :param module_size: variable controlling total amount of latent module topology parameters.
        :param split: variable sets y dimension of latent module topology.
        :param gpu: the gpu index.
        :param module_type: value indicator to identify between generator(1) and discriminator(2) modules.
        :return: This function returns the generated module topology array and forms the topology parameters.
        """
        start, factor, gpu_ops, counter, combined_factor, saved_concat = time.time(), dict(), 0, 0, 0, 0
        image_size_x, image_size_y, scaling = 8 if module_type == 1 and not self.args['indir'] else self.growth['x_size'], 8 if module_type == 1 and not self.args['indir'] else self.growth['y_size'], self.growth['scaling']
        split = split[0] if type(split) is list else split[0].tolist()
        parameters = np.split(parameters, module_size, 1)
        saved_image_size_x, saved_image_size_y, saved_total_filter, max_filter_size = image_size_x, image_size_y, self.specifications['total_filters'], self.specifications['max_filter_size']
        for i in range(len(split)):
            force_stride, stop_stride, stride_check, saved_image_size_x, saved_image_size_y = True if scaling >= (len(split) - (i + 1)) and module_type == 1 else False, False, 0, image_size_x, image_size_y
            for ii in range(split[i]):
                stride_check = (math.ceil((parameters[(counter * 3) + (ii * 3) + 2])[0].item() * (self.specifications['max_stride_size']))) if (stride_check != 2 and not force_stride) else 2
                if (stride_check == 2 and module_type == 1 and saved_image_size_x * 2 > self.growth['x_size']) or self.args['indir']:
                    stop_stride, stride_check = True, 1
            for ii in range(split[i]):
                remaining_split_filters = max_filter_size if saved_total_filter > max_filter_size else saved_total_filter
                split_max_filter = remaining_split_filters if remaining_split_filters > combined_factor else remaining_split_filters - combined_factor
                factor['filter_%d_%d_%d' % (ii, i, gpu)] = math.ceil((parameters[(counter * 3) + (ii * 3) + 0])[0].item() * split_max_filter)
                factor['filter_%d_%d_%d' % (ii, i, gpu)] = 1 if 0 >= factor['filter_%d_%d_%d' % (ii, i, gpu)] else factor['filter_%d_%d_%d' % (ii, i, gpu)]
                saved_total_filter -= factor['filter_%d_%d_%d' % (ii, i, gpu)] if (saved_total_filter - factor['filter_%d_%d_%d' % (ii, i, gpu)]) != 0 else 0
                combined_factor += -combined_factor if ii == 0 else factor['filter_%d_%d_%d' % (ii, i, gpu)]
                factor['kernel_%d_%d_%d' % (ii, i, gpu)] = math.ceil((parameters[(counter * 3) + (ii * 3) + 1])[0].item() * (self.specifications['max_kernel_size']))
                factor['kernel_%d_%d_%d' % (ii, i, gpu)] = 1 if factor['kernel_%d_%d_%d' % (ii, i, gpu)] == 0 else factor['kernel_%d_%d_%d' % (ii, i, gpu)]
                factor['stride_%d_%d_%d' % (ii, i, gpu)] = (math.ceil((parameters[(counter * 3) + (ii * 3) + 2])[0].item() * (self.specifications['max_stride_size']))) if stride_check != 2 else 2
                factor['stride_%d_%d_%d' % (ii, i, gpu)] = 1 if factor['stride_%d_%d_%d' % (ii, i, gpu)] == 0 or stop_stride else factor['stride_%d_%d_%d' % (ii, i, gpu)]
                if stride_check == 2 and ii == 0 and not stop_stride:
                    if module_type == 1:
                        image_size_x *= 2
                        image_size_y *= 2
                        scaling -= 1
                    else:
                        image_size_x /= 2
                        image_size_y /= 2
                image_size_y, image_size_x = int(image_size_y) if image_size_y > 1 else 1, int(image_size_x) if image_size_x > 1 else 1
                gpu_ops += factor['kernel_%d_%d_%d' % (ii, i, gpu)] * factor['kernel_%d_%d_%d' % (ii, i, gpu)] * (saved_concat if i > 0 else 3) * factor['filter_%d_%d_%d' % (ii, i, gpu)] * 3 * 4
                gpu_ops += factor['filter_%d_%d_%d' % (ii, i, gpu)] * image_size_x * image_size_y * self.growth["batch_size"] * 2 * 2 * 4
                gpu_ops += saved_concat * saved_image_size_x * saved_image_size_y * self.growth["batch_size"] * 2 * 4
                saved_concat = combined_factor if split[i] == ii + 1 else saved_concat
            counter += split[i]
        factor['kernel_%d' % gpu] = math.ceil((parameters[counter + 1])[0].item() * (self.specifications['max_kernel_size']))
        factor['kernel_%d' % gpu] = 1 if factor['kernel_%d' % gpu] == 0 else factor['kernel_%d' % gpu]
        factor['filter_%d' % gpu] = math.ceil((parameters[counter + 2])[0].item() * (max_filter_size if saved_total_filter > max_filter_size else saved_total_filter))
        factor['filter_%d' % gpu] = (3 * self.args['num_comp']) if module_type == 1 else ((1 if factor['filter_%d' % gpu] == 0 else factor['filter_%d' % gpu]) * self.args['num_comp'])
        gpu_ops += factor['kernel_%d' % gpu] * factor['kernel_%d' % gpu] * saved_concat * factor['filter_%d' % gpu] * 3 * 4
        gpu_ops += factor['filter_%d' % gpu] * (self.growth['x_size'] if module_type == 1 else image_size_x) * (self.growth['y_size'] if module_type == 1 else image_size_y) * self.growth['batch_size'] * 2 * (2 if module_type == 1 else 1) * 4
        gpu_ops += saved_concat * (self.growth['x_size'] if module_type == 1 else saved_image_size_x) * (self.growth['y_size'] if module_type == 1 else saved_image_size_y) * self.growth['batch_size'] * 2 * 4
        performance_tracker(start, '%s parameters completed in %fs', "Generator module %d" % gpu if module_type == 1 else "Discriminator module %d" % gpu)
        return factor, gpu_ops

    def prepare_graph(self, module_type, struct, stage, gpu):
        """
        :param module_type: value indicator to identify between generator(1) and discriminator(2) modules.
        :param struct: dictionary of topology parameters to be fed to generator and discriminator modules.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :param gpu: the gpu index.
        :return: This function returns the struct dictionary of created latent module topologies.
        """
        start, gpu_ops, live_ops = time.time(), 0, None
        total_split, total_parameters = self.template(module_type, struct, gpu)
        struct['disc_iter'] = 2 if module_type == 2 else 0
        feed = total_split + total_parameters
        module_feed = np.array(feed, dtype=np.float32).reshape(1, (self.specifications['max_module_size'] * 2))
        iter_feed = np.array(struct['disc_iter']).reshape(1, 1)
        struct['teacher_feed_%d_%d' % (module_type, gpu)] = np.concatenate([module_feed, iter_feed], 1)
        self.completing('generator' if module_type == 1 else 'discriminator', module_type, struct, gpu)
        gpu_ops += struct['%s_ops_%d' % ('generator' if module_type == 1 else 'discriminator', gpu)] * (2 if module_type == 1 else 3)
        if stage == 3 and module_type == 2:
            np.save(os.path.join(self.location, 'struct.npy'), struct)
        if stage == 3 or stage == 4:
            teacher = tf.placeholder(tf.float32, [None, self.size], name='teacher_ph')
            _, _, live_ops = self.living(teacher, struct, stage)
            score = self.module_estimator(teacher, module_type, live_ops, struct, gpu)
        else:
            score = 1
        performance_tracker(start, 'Latent topology generator graph completed in %fs')
        return gpu_ops, struct, score

    def preparing(self, stage):
        """
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function returns a batch of graphs that fit the memory of each gpu in use.
        """
        start = time.time()
        total_ops, struct = 0, dict()
        for gpu in range(self.specifications['gpus'] if stage == 2 else 1):
            target_score, pred_score, module_type, safety = 0 if stage == 2 else self.growth['model_target'], 0, 1, 0
            while True:
                if (total_ops > ((self.specifications['max_memory_%d' % gpu] * 0.75) // 2)) or (target_score > float(pred_score)) or module_type != 2:
                    tf.reset_default_graph()
                    total_ops, struct, pred_score = self.prepare_graph(module_type, struct, stage, gpu)
                    if self.args['console'] and stage == 3:
                        print("Module %d Score: %d/%d Memory: %d/%d" % (module_type, pred_score, target_score, total_ops, ((self.specifications['max_memory_%d' % gpu] * 0.75) // 2)))
                    if ((self.specifications['max_memory_%d' % gpu] * 0.75) // 2) >= total_ops and float(pred_score) >= target_score and module_type != 2:
                        module_type, total_ops = 2, ((self.specifications['max_memory_%d' % gpu] * 0.75) // 2) + 1
                    safety += 1
                    target_score -= 5 if safety == 100 else 0
                    safety = safety if safety != 100 else 0
                else:
                    logging.info('Generator %d: %dops' % (gpu + 1, struct['generator_ops_%d' % gpu]) + str(struct['generator_factor_%d' % gpu]))
                    logging.info('Discriminator %d: %dops' % (gpu + 1, struct['discriminator_ops_%d' % gpu]) + str(struct['discriminator_factor_%d' % gpu]))
                    logging.info('Ops %d: %dops/%dops' % (gpu + 1, struct['generator_ops_%d' % gpu] + struct['discriminator_ops_%d' % gpu], (self.specifications['max_memory_%d' % gpu] * 0.75)))
                    first_loop, total_ops, pred_score = True, 0, 0
                    break
        tf.reset_default_graph()
        performance_tracker(start, 'Setting topologies completed in %fs')
        return struct, total_ops

    def infer_iicc(self, eval_feed, struct, stage):
        """
        :param eval_feed: dictionary of pairs of ground truth images and generated images.
        :param struct: dictionary of topology parameters to be fed to generator and discriminator modules.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function returns a classification score for all the generated images, using the IICC saved model.
        """
        tf.reset_default_graph()
        iicc_learning = tf.placeholder(tf.bool, name='iicc_learning')
        ground = multi_placeholders(self.args['num_comp'], 'ground_truth_images')
        noise = multi_placeholders(self.args['num_comp'], 'noise_images')
        augmented = multi_placeholders(self.args['num_comp'], 'augmented_images')
        true_pair, fake_pair, noise_pair = self.iicc.cluster(ground, iicc_learning, self.growth['small_x_size']), self.iicc.cluster(augmented, iicc_learning, self.growth['small_x_size']), self.iicc.cluster(noise, iicc_learning, self.growth['small_x_size'])
        true_logits, fake_logits, noise_logits = [tf.argmax(self.iicc.classifier_out(true_pair, 20, 'over' + str(i + 1)), axis=1) for i in range(5)], [tf.argmax(self.iicc.classifier_out(fake_pair, 20, 'over' + str(i + 1)), axis=1) for i in range(5)], [tf.argmax(self.iicc.classifier_out(noise_pair, 20, 'over' + str(i + 1)), axis=1) for i in range(5)]
        with tf.Session(config=tf_config_setup()) as sess:
            start = time.time()
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            performance_tracker(start, 'Initialization completed in %fs')
            start = time.time()
            learn_variables = [var for var in tf.global_variables() if var.name.startswith('learn/') or var.name.startswith('true1/')]
            learn = tf.train.Saver(learn_variables)
            learn.restore(sess, os.path.join(os.path.join(self.location, 'iicc'), 'learn.ckpt'))
            performance_tracker(start, 'Saver completed in %fs')
            for gpu in range(self.specifications['gpus']):
                if not struct['nan_found']:
                    gpu_feed, eval_list, noise_list = {iicc_learning: False}, [], []
                    for i in range(self.args['num_comp']):
                        eval_list.append((eval_feed['output_%d' % gpu][i] if self.args['num_comp'] > 1 else eval_feed['output_%d' % gpu]) + 1.0 / 2.0)
                        noise_list.append((np.random.rand(16, self.growth['small_y_size'], self.growth['small_x_size'], 3).astype(np.float32) * 255) + 1.0 / 2.0)
                    main_feed = multi_dict_feed(self.args, gpu_feed, augmented, eval_list if self.args['num_comp'] > 1 else eval_list[0], ground, eval_feed['comp_%d' % gpu], '2_end')
                    noise_feed = multi_dict_feed(self.args, gpu_feed, noise, noise_list if self.args['num_comp'] > 1 else noise_list[0], ground, eval_feed['comp_%d' % gpu], '2_end')
                    start = time.time()
                    true_hat, fake_hat = sess.run([true_logits, fake_logits], main_feed)
                    noise_hat = sess.run(noise_logits, noise_feed)
                    performance_tracker(start, 'Reflection %d session completed in %fs', gpu)
                    start, hat_score = time.time(), 0
                    for i in range(5):
                        true_val, fake_val, noise_val = set(sorted(true_hat[i])), set(sorted(fake_hat[i])), set(sorted(noise_hat[i]))
                        hat_check = true_val & fake_val
                        noise_check = noise_val & fake_val
                        hat_score += (2 * len(hat_check)) / len(fake_val)
                        hat_score -= (4 * len(noise_check)) / len(fake_val)
                    learn_score = hat_score / 5
                else:
                    learn_score = 0
                gpu_score = (0.0 if 0 > (learn_score * 100) else learn_score) if 100 > (learn_score * 100) else 1.0
                self.growth['teacher_feeds'].append([gpu_score, struct['teacher_feed_1_%d' % gpu]])
                self.growth['teacher_feeds'].append([gpu_score, struct['teacher_feed_2_%d' % gpu]])
                self.growth['model_quality'].append(int(gpu_score * 100))
                self.specifications['epoch_count'] += 1
                if not os.path.exists(os.path.join(self.location, 'progress')):
                    os.makedirs(os.path.join(self.location, 'progress'))
                self.infer_images(eval_feed, gpu, stage)
                if self.args['console']:
                    print("Model %d/%d score: " % (self.specifications['epoch_count'], self.specifications['total_epoch_count']) + str(gpu_score))
                performance_tracker(start, 'Result %d session completed in %fs', gpu)
            return

    def infer_images(self, eval_feed, gpu, stage):
        """
        :param eval_feed: dictionary of pairs of ground truth images and generated images.
        :param gpu: the gpu index.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function saves the inferred imaged.
        """
        for i in range(self.growth['batch_size']):
            for ii in range(self.args['num_comp']):
                imageio.imwrite(os.path.join(os.path.join(self.location, 'progress' if stage != 4 else 'draw'), ('model_output_%d_%d_%d.png' % (self.specifications['epoch_count'], i + 1, ii + 1)) if stage != 4 else ('final_output_%d_%d.png' % (i + 1, ii + 1))), ((eval_feed['output_%d' % gpu][ii][i]*255).astype(np.uint8)) if self.args['num_comp'] > 1 else ((eval_feed['output_0'][i]*255).astype(np.uint8)))
                if self.args['console']:
                    print("Generated image %d of %d has been saved to " % (i + 1, ii + 1) + str(os.path.join(os.path.join(self.location, 'progress' if stage != 4 else 'draw'), ('model_output_%d_%d_%d.png' % (self.specifications['epoch_count'], i + 1, ii + 1)) if stage != 4 else ('final_output_%d_%d.png' % (i + 1, ii + 1)))))
        return

    def train_estimator(self, stage):
        """
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function returns a trained module estimator and an updated index.
        """
        tf.reset_default_graph()
        teacher = tf.placeholder(tf.float32, [None, self.size], name='teacher_ph')
        quality = tf.placeholder(tf.float32, [None, 1], name='quality_ph')
        construct, loss, estimated = self.living(teacher, stage, quality)
        with tf.Session(config=tf_config_setup()) as sess:
            coord, threads = initialization(sess)
            start = time.time()
            variables_to_restore = [var for var in tf.global_variables() if var.name.startswith('weights_') or var.name.startswith('bias_')]
            live = tf.train.Saver(variables_to_restore)
            performance_tracker(start, 'Saver completed in %fs')
            self.growth['model_target'] = calculate_target(self.growth['model_quality'])
            feed_copied = self.growth['teacher_feeds'].copy()
            for index, value in enumerate(feed_copied):
                for i in range(10 if value[0]*100 > self.growth['model_target'] else 0):
                    augmenting = (list(feed_copied[index][1][0] * (0.55 + (i * 0.1))))
                    augmented_feed = np.array(augmenting).reshape(1, (self.specifications['max_module_size'] * 2) + 1)
                    self.growth['teacher_feeds'].append([feed_copied[index][0], augmented_feed])
            for epoch in range(self.growth['estimator_length']):
                quality_mini_batch, teacher_mini_batch = setup_mini_batch(self.growth['teacher_feeds'], self.specifications)
                epoch_feed = {quality: quality_mini_batch, teacher: teacher_mini_batch}
                start = time.time()
                _, current_loss, pred = sess.run([construct, loss, estimated], epoch_feed)
                performance_tracker(start, 'Shaping session completed in %fs')
                if epoch % (self.growth['estimator_length'] / 100) == 0:
                    if self.args['console']:
                        print("Pred: " + str(pred))
                        print("Real: " + str(quality_mini_batch))
                        print("Epoch: %d " % epoch + "Module estimator loss: " + str(current_loss))
            start = time.time()
            live.save(sess, os.path.join(self.location, 'live.ckpt'))
            performance_tracker(start, 'Saving discovery completed in %fs')
            self.specifications['live_learning'] = False
            coord.request_stop()
            coord.join(threads)
            return

    def saver(self, sess, stage):
        """
        :param sess: tensorflow active session.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function returns the stage tf.train.Saver() and restores previous saved weights.
        """
        start = time.time()
        first_variables = [var for var in tf.global_variables() if var.name.startswith('learn/conv1') or var.name.startswith('learn/norm1') or var.name.startswith('first1/first1dense')]
        first_learned = tf.train.Saver(first_variables)
        draw, learn = None, None
        if stage == 2 or stage == 3:
            first_learned.restore(sess, os.path.join(os.path.join(self.location, 'iicc'), 'first_learned.ckpt'))
        if stage != 2:
            draw, learn = tf.train.Saver() if stage != 1 else None, tf.train.Saver() if stage == 1 else None
            if (stage == 3 and self.growth['saved_epoch'] != 0) or stage == 4:
                draw.restore(sess, os.path.join(os.path.join(self.location, 'draw'), 'draw.ckpt'))
        performance_tracker(start, 'Saver completed in %fs')
        return draw, learn, first_learned

    def stage_ops(self, sess, epoch, eval_feed, epochs_truth, struct, graph, models, epoch_feed, total_epochs, stage):
        """
        :param sess: tensorflow active session.
        :param epoch: current epoch in stage loop.
        :param eval_feed: dictionary of pairs of ground truth images and generated images.
        :param epochs_truth: ground truth image batch.
        :param struct: dictionary of topology parameters to be fed to generator and discriminator modules.
        :param graph: dictionary of all graph elements.
        :param models: dictionary of all model elements.
        :param epoch_feed: feed_dict to session.
        :param total_epochs: total epochs in stage loop.
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function runs the stage's training/inference.
        """
        start = time.time()
        struct['nan_found'] = False
        if stage == 1:
            _, _, loss = sess.run([graph['over_ops'], graph['true_ops'], models['losses']], epoch_feed)
            if self.args['console']:
                print("Epoch: %d " % epoch + "IICC loss: " + str(loss))
        if stage == 2 or stage == 3:
            for gpu in range(self.specifications['gpus'] if stage == 2 else 1):
                magic_loss, magic_output_loss = sess.run([graph['magic_loss_%d' % gpu], graph['magic_output_loss_%d' % gpu]], epoch_feed)
                if self.args['console']:
                    print("Epoch: %d " % epoch + "Discriminator %d model loss: " % (gpu + self.specifications['epoch_count'] + 1) + str(magic_loss), "Generator %d model loss: " % (gpu + self.specifications['epoch_count'] + 1) + str(magic_output_loss) + ")")
                if epoch != 0:
                    if math.isnan(magic_loss) or math.isnan(magic_output_loss):
                        struct['nan_found'], epoch = True, total_epochs
                        break
            for _ in range(struct['disc_iter']):
                sess.run(graph['disc'], epoch_feed)
            generated_output, _ = sess.run([models, graph['gen']], epoch_feed)
            for gpu in range(self.specifications['gpus'] if stage == 2 else 1):
                eval_feed['output_%d' % gpu], eval_feed['comp_%d' % gpu] = generated_output['magic_output_%d' % gpu], epochs_truth
        if stage == 4:
            output = sess.run(models, epoch_feed)
            eval_feed['output_0'] = output['magic_output_0']
        performance_tracker(start, 'Session %d completed in %fs', epoch)
        return struct['nan_found'], epoch

    def stage_save(self, epoch, sess, draw, learn, first_learned, stage):
        """
        :param epoch: current epoch in stage loop.
        :param sess: tensorflow active session.
        :param draw: stage 3 tf.train.saver().
        :param learn: all variables stage 1 tf.train.saver().
        :param first_learned: first layer variables stage 1 tf.train.saver()
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return:
        """
        start = time.time()
        if stage == 1:
            learn.save(sess, os.path.join(os.path.join(self.location, 'iicc'), 'learn.ckpt'))
            first_learned.save(sess, os.path.join(os.path.join(self.location, 'iicc'), 'first_learned.ckpt'))
            output_node_names = ['true1/true1dense/Softmax', 'true2/true2dense/Softmax', 'true3/true3dense/Softmax', 'true4/true4dense/Softmax', 'true5/true5dense/Softmax']
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
            with open(os.path.join(self.location, 'iicc.pb'), 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())
        if stage == 3:
            draw.save(sess, os.path.join(os.path.join(self.location, 'draw'), 'draw.ckpt'))
            try:
                output_node_names = ['generator_model_memory_0/Tanh']
                frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
                with open(os.path.join(self.location, 'draw.pb'), 'wb') as f:
                    f.write(frozen_graph_def.SerializeToString())
            except ValueError:
                logging.error("The graph cannot be frozen as a .pb file since a tensor larger than 2gb has been found.")
            self.growth['saved_epoch'] = epoch
            np.save(os.path.join(os.path.join(self.location, 'draw'), 'growth.npy'), self.growth)
        performance_tracker(start, 'Saving session %d completed in %fs', epoch)
        return

    def start(self, stage):
        """
        :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
        :return: This function runs the convstruct function.
        """
        main_start = time.time()
        create_log(self.location, 'draw.log' if stage == 3 or stage == 4 else ('live.log' if stage == 2 else 'learn.log'))
        struct, total_ops = self.preparing(stage) if stage == 2 or (stage == 3 and self.growth['saved_epoch'] == 0) else (dict() if stage == 1 else np.load(os.path.join(self.location, 'struct.npy')).flat[0], None)
        get_ground_truth_batch = build_inputs(self.args, self.args['compdir'], self.args['num_comp'], self.growth, stage)
        get_starting_point_batch = build_inputs(self.args, self.args['indir'], self.args['num_in'], self.growth, stage) if self.args['indir'] else None
        iicc_learning = tf.placeholder(tf.bool, name='iicc_learning')
        truth = multi_placeholders(self.args['num_comp'], 'ground_truth')
        augmented = multi_placeholders(self.args['num_comp'], 'augmented')
        graph, models, epoch_feed = self.active_graph(truth, iicc_learning, struct, stage) if stage != 1 else self.iicc.clustering(augmented, truth, iicc_learning, self.growth)
        with tf.Session(config=tf_config_setup()) as sess:
            coord, threads = initialization(sess)
            draw, learn, first_learned = self.saver(sess, stage)
            total_epochs, epoch, eval_feed = ((self.growth['initial_length'] if stage == 2 else self.growth['full_length']) if stage != 1 else self.growth['iicc_length']) if stage != 4 else 1, self.growth['saved_epoch'] if stage != 4 else 0, dict()
            while True:
                if epoch < total_epochs:
                    epochs_ground_truth, epochs_starting_points, epochs_augmented = setup_batch(self.args, sess, get_ground_truth_batch, get_starting_point_batch, self.growth, epoch if stage == 1 else None)
                    epochs_starting_points, epochs_ground_truth, epochs_augmented = preprocess(self.args, epochs_starting_points, epochs_ground_truth, epochs_augmented, stage)
                    epoch_feed = multi_dict_feed(self.args, epoch_feed, augmented if stage == 1 else graph['starting_points'], epochs_augmented if stage == 1 else epochs_starting_points, truth, epochs_ground_truth, stage)
                    struct['nan_found'], epoch = self.stage_ops(sess, epoch, eval_feed, epochs_ground_truth, struct, graph, models, epoch_feed, total_epochs, stage)
                    if struct['nan_found']:
                        break
                    if epoch == (total_epochs - 1) or (epoch % 100 == 0 and epoch != 0 and stage == 3):
                        self.stage_save(epoch, sess, draw, learn, first_learned, stage)
                    epoch += 1
                else:
                    break
            coord.request_stop()
            coord.join(threads)
        if stage == 2:
            self.infer_iicc(eval_feed, struct, stage)
        if stage == 2 and self.specifications['epoch_count'] >= self.specifications['total_epoch_count']:
            self.train_estimator(stage)
        if stage == 3:
            self.growth['draw_learning'] = False
        if stage == 4:
            self.infer_images(eval_feed, 0, stage)
        np.save(os.path.join(self.location, 'specifications.npy'), self.specifications)
        np.save(os.path.join((self.location if stage != 3 else os.path.join(self.location, 'draw')), 'growth.npy'), self.growth)
        performance_tracker(main_start, ('Live loop completed in %fs' if stage != 3 else 'Draw completed in %fs') if stage != 1 else 'Learn completed in %fs')
        return
