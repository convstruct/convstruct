from .util import *


class Graphing:
    def __init__(self, args, gpu, memory, learning, factor, growth, split, stage, strength=10):
        """
        :param args: dict, dictionary of arguments.
        :param gpu: int, the gpu index.
        :param memory: str, changing string for the tf.variable scope of the current module.
        :param learning: tf.bool, boolean for enabling or disabling batch normalization.
        :param factor: dict, dictionary holding all module/model topology parameters.
        :param growth: dict, dictionary of info on input, output data and progress.
        :param split: int list, shape of module topology.
        :param stage: int, value indicator to identify between learn(1), live(2), and draw(3/4).
        :param strength: value to set strength of the gradient penalty.
        """
        self.args = args
        self.gpu = gpu
        self.memory = memory
        self.learning = learning
        self.factor = factor
        self.split = split
        self.growth = growth
        self.stage = stage
        self.strength = strength
        self.s_dims = 8 if stage != 2 else 2
        self.y_dim = growth['y_size'] if stage != 2 or args['indir'] else growth['small_y_size']
        self.x_dim = growth['x_size'] if stage != 2 or args['indir'] else growth['small_x_size']

    def createGraph(self, magic_input, num_feed, discriminator=None):
        """
        :param magic_input: tf.placeholder, input data into a topology.
        :param num_feed: int, represents number of inputs or number of outputs of a topology.
        :param discriminator: int, optional, value to indicate if the model is a generator (None) or a discriminator (Any).
        :return: This function returns the high level topology for generators and discriminators.
        """
        def module(args, gpu, module_input, learning, split, factor):
            """
            :param args: dict, dictionary of arguments.
            :param gpu: int, the gpu index.
            :param module_input: tf.placeholder, input data to the module.
            :param learning: tf.bool, boolean for enabling or disabling batch normalization.
            :param split: int list, shape of module topology.
            :param factor: dict, dictionary holding all module/model topology parameters.
            :return: This function returns the module topology for generators and discriminators.
            """
            conv_type, layers, activation, multi_output = tf.layers.conv2d if discriminator is not None else tf.layers.conv2d_transpose, dict(), tf.nn.leaky_relu if discriminator is not None else tf.nn.relu, []
            for i in range(len(split)):
                combine = []
                for ii in range(split[i]):
                    layers['module_input_%d_%d' % (ii, i)] = conv_type(layers['module_input_0_%d' % (i - 1)] if i != 0 else module_input, filters=factor['filter_%d_%d_%d' % (ii, i, gpu)], kernel_size=factor['kernel_%d_%d_%d' % (ii, i, gpu)], strides=factor['stride_%d_%d_%d' % (ii, i, gpu)], padding="same")
                    layers['module_input_%d_%d' % (ii, i)] = tf.layers.batch_normalization(layers['module_input_%d_%d' % (ii, i)], training=learning)
                    layers['module_input_%d_%d' % (ii, i)] = activation(layers['module_input_%d_%d' % (ii, i)])
                    if split[i] > 1:
                        combine.append(layers['module_input_%d_%d' % (ii, i)])
                        if split[i] == ii + 1 and len(split) != i + 1:
                            layers['module_input_0_%d' % i] = tf.concat(combine, 3)
                        elif split[i] == ii + 1 and len(split) == i + 1:
                            module_input = tf.concat(combine, 3)
                    else:
                        module_input = layers['module_input_%d_%d' % (ii, i)]
            module_input = conv_type(module_input, filters=factor['filter_%d' % gpu], kernel_size=factor['kernel_%d' % gpu], strides=1, padding="same")
            module_input = tf.layers.batch_normalization(module_input, training=learning) if discriminator is not None else module_input
            module_output = tf.nn.leaky_relu(module_input) if discriminator is not None else tf.nn.tanh(module_input)
            if args['num_comp'] > 1:
                module_output = tf.split(module_output, num_or_size_splits=args['num_comp'], axis=3)
            if discriminator is not None:
                for num in range(args['num_comp']):
                    dim = int(np.prod(module_output[num].get_shape()[1:])) if args['num_comp'] > 1 else int(np.prod(module_output.get_shape()[1:]))
                    flat = tf.reshape(module_output[num], shape=[-1, dim], name='flat') if args['num_comp'] > 1 else tf.reshape(module_output, shape=[-1, dim], name='flat')
                    weight = tf.get_variable('w', shape=[dim, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
                    bias = tf.get_variable('b', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
                    if args['num_comp'] > 1:
                        multi_output.append(tf.add(tf.matmul(flat, weight), bias, name='y'))
                        module_output = multi_output if num + 1 == args['num_comp'] else module_output
                    else:
                        module_output = tf.add(tf.matmul(flat, weight), bias, name='y')
            return module_output

        with tf.variable_scope(self.memory, reuse=tf.AUTO_REUSE):
            if num_feed > 1:
                magic_input = tf.concat(list(magic_input.values()), 1 if discriminator is None and not self.args['indir'] else 3) if type(magic_input) is dict else tf.concat(magic_input, 1 if discriminator is None and not self.args['indir'] else 3)
            if discriminator is None and not self.args['indir']:
                magic_input.set_shape([None, (100 * self.args['num_in'])])
                w = tf.get_variable('w', shape=[(100 * self.args['num_in']) if not self.args['indir'] else (10 * self.args['num_in'] * 10 * self.args['num_in']), self.s_dims * self.s_dims * 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('b', shape=[self.s_dims * self.s_dims * 512], dtype=tf.float32, initializer=tf.zeros_initializer())
                flat_conv = tf.add(tf.matmul(magic_input, w), b, name='flat_conv')
                conv = tf.reshape(flat_conv, shape=[-1, self.s_dims, self.s_dims, 512], name='conv')
                magic_input = tf.layers.batch_normalization(conv, training=self.learning)
                magic_input = tf.nn.relu(magic_input)
            else:
                magic_input.set_shape([None, self.y_dim, self.x_dim, (3 * num_feed)])
            magic_output = module(self.args, self.gpu, magic_input, self.learning, self.split, self.factor)
            return magic_output

    def createLoss(self, generated_logits, ground_logits=None, sum_hat=None, sum_logits=None, true_hat=None, fake_hat=None):
        """
        :param generated_logits: float array, generator output's logits from discriminator.
        :param ground_logits: float array, optional, ground truth logits from discriminator.
        :param sum_hat: float array, optional, sum of ground and generated output.
        :param sum_logits: float array, optional, logits of the sum of ground and generated output.
        :param true_hat: float array, optional, logits from comparison data classified using convstruct.learn().
        :param fake_hat: float array, optional, logits from generated data classified using convstruct.learn().
        :return: This function returns the loss of a module.
        """
        if ground_logits is None:
            total_loss = -tf.reduce_mean(generated_logits)
        else:
            gradient_penalty = tf.gradients(sum_logits, sum_hat)[0]
            gradient_penalty = tf.sqrt(tf.reduce_sum(tf.square(gradient_penalty), axis=1))
            gradient_penalty = tf.reduce_mean(tf.square(gradient_penalty - 1.0) * self.strength)
            iicc2 = tf.reduce_mean(tf.square(true_hat - fake_hat))
            gan = tf.reduce_mean(generated_logits) - tf.reduce_mean(ground_logits)
            total_loss = gan + gradient_penalty + iicc2
        return total_loss

    def createOptimizer(self, loss):
        """
        :param loss: tf.reduce_mean(), the module loss.
        :return: This function returns the optimizer of a module.
        """
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.memory)
        with tf.control_dependencies(ops):
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.memory)
            return tf.train.AdamOptimizer(learning_rate=0.00015, beta1=0.5, beta2=0.9).minimize(loss, var_list=var)
