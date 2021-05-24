from .util import *


class IICC:
    def __init__(self, args, location, memory, repeats=5, bias_init=tf.constant_initializer(0.0), kernel_init=tf.initializers.variance_scaling(dtype=tf.float32)):
        """
        :param args: dict, dictionary of arguments.
        :param location: str, path of directory to save weights and summaries to.
        :param memory: str, changing string for the tf.variable scope of the classifier.
        :param repeats: int, optional, number of classifiers.
        :param bias_init: tf.constant_initializer(), optional, initializer of classifier bias.
        :param kernel_init: tf.initializers.variance_scaling(), optional, initializer of classifier kernel.
        """
        self.args = args
        self.location = location
        self.memory = memory
        self.repeats = repeats
        self.bias_init = bias_init
        self.kernel_init = kernel_init

    def firstLayers(self, remember, learning, dim_size):
        """
        :param remember: tf.placeholder, input data to classify.
        :param learning: tf.bool, boolean for enabling or disabling batch normalization.
        :param dim_size: int, dimension of input and output data.
        :return: This function returns the topology for the first layer of the classifier.
        """
        remember = (tf.concat(list(remember.values()), 3) if type(remember) is dict else tf.concat(remember, 3)) if self.args['num_comp'] > 1 else remember
        remember = tf.image.resize(remember, [dim_size, dim_size])
        remember.set_shape([None, dim_size, dim_size, (3 * self.args['num_comp']) if self.args['num_comp'] > 1 else 3])
        remembered = tf.layers.conv2d(remember, filters=64, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu, name="conv1")
        remembered = tf.layers.batch_normalization(remembered, training=learning, name="norm1")
        remembered = tf.layers.max_pooling2d(remembered, pool_size=2, strides=2, padding="SAME")
        remembered = tf.layers.conv2d(remembered, filters=128, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
        remembered = tf.layers.batch_normalization(remembered, training=learning)
        remembered = tf.layers.max_pooling2d(remembered, pool_size=2, strides=2, padding="SAME")
        return remembered

    def getFirstLayers(self, remember, learning, growth):
        """
        :param remember: tf.placeholder, input data to classify.
        :param learning: tf.bool, boolean for enabling or disabling batch normalization.
        :param growth: dict, dictionary of info on input and output data.
        :return: This function returns the topology for the small cluster.
        """
        with tf.variable_scope(self.memory, reuse=tf.AUTO_REUSE):
            early = self.firstLayers(remember, learning, growth)
        return early

    def getFullClassifier(self, remember, learning, dim_size):
        """
        :param remember: tf.placeholder, input data to classify.
        :param learning: tf.bool, boolean for enabling or disabling batch normalization.
        :param dim_size: int, dimension of input and output data.
        :return: This function returns the topology for the full cluster.
        """
        with tf.variable_scope(self.memory, reuse=tf.AUTO_REUSE):
            remembered = self.firstLayers(remember, learning, dim_size)
            remembered = tf.layers.conv2d(remembered, filters=256, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
            remembered = tf.layers.batch_normalization(remembered, training=learning)
            remembered = tf.layers.max_pooling2d(remembered, pool_size=2, strides=2, padding="SAME")
            remembered = tf.layers.conv2d(remembered, filters=512, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
            remembered = tf.layers.batch_normalization(remembered, training=learning)
            remembered = tf.layers.max_pooling2d(remembered, pool_size=2, strides=2, padding="SAME")
            remembered = tf.layers.flatten(remembered)
            return remembered

    def getClassifierLogits(self, input_pair, num_clusters, name):
        """
        :param input_pair: tf.placeholder, the part of a pair from the ground truth images.
        :param num_clusters: int, the number of clusters to classify images into.
        :param name: str, identity of the current cluster.
        :return: This function returns the logits of the pairing from the classifier's softmax output.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(inputs=input_pair, units=num_clusters, activation=tf.nn.softmax, use_bias=True, kernel_initializer=self.kernel_init, bias_initializer=self.bias_init, name=(name + 'dense'))
        return logits

    def getClassifierLoss(self, true_pair, aug_pair, num_clusters, num_heads, head):
        """
        :param true_pair: tf.placeholder, the ground truth image.
        :param aug_pair: tf.placeholder, an augmentation of the ground truth image.
        :param num_clusters: int, the number of clusters to classify images into.
        :param num_heads: int, the number of clustering attempts.
        :param head: str, identity of the current cluster.
        :return: This function returns the loss function of the classifier.
        """
        loss = tf.constant(0, dtype=tf.float32)
        for i in range(num_heads):
            true_logits = self.getClassifierLogits(true_pair, num_clusters, name=head + str(i + 1))
            aug_logits = self.getClassifierLogits(aug_pair, num_clusters, name=head + str(i + 1))
            true_shape = true_logits.shape.as_list()[1]
            combined_logits = tf.transpose(true_logits) @ aug_logits
            averaged_logits = (combined_logits + tf.transpose(combined_logits)) / 2
            head_logits = tf.clip_by_value(averaged_logits, clip_value_min=1e-6, clip_value_max=tf.float32.max)
            head_logits /= tf.reduce_sum(head_logits)
            head_logits_x = tf.broadcast_to(tf.reshape(tf.reduce_sum(head_logits, axis=0), (true_shape, 1)), (true_shape, true_shape))
            head_logits_y = tf.broadcast_to(tf.reshape(tf.reduce_sum(head_logits, axis=1), (1, true_shape)), (true_shape, true_shape))
            loss += -tf.reduce_sum(head_logits * (tf.math.log(head_logits) - tf.math.log(head_logits_x) - tf.math.log(head_logits_y)))
        loss /= num_heads
        return loss

    def getOptimizer(self, loss):
        """
        :param loss: tf.reduce_sum(), the classifier loss.
        :return: This function returns the optimizer of the classifier.
        """
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.memory)
        with tf.control_dependencies(ops):
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.memory)
            return tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var)

    def getGraph(self, augmented, ground, iicc_learning, growth):
        """
        :param augmented: tf.placeholder, augmented pair created from ground truth image.
        :param ground: tf.placeholder, ground truth image.
        :param iicc_learning: tf.bool, boolean for enabling or disabling batch normalization.
        :param growth: dict, dictionary of info on input and output data.
        :return: This function graphs the IICC classifier and calculates and optimizes its loss.
        """
        start = time.time()
        graph, models, epoch_feed = dict(), dict(), {iicc_learning: True}
        true_pair = self.getFullClassifier(ground, iicc_learning, growth['x_size'])
        aug_pair = self.getFullClassifier(augmented, iicc_learning, growth['x_size'])
        small_true_pair = self.getFullClassifier(ground, iicc_learning, growth['small_x_size'])
        small_aug_pair = self.getFullClassifier(augmented, iicc_learning, growth['small_x_size'])
        loss_a = self.getClassifierLoss(small_true_pair, small_aug_pair, 20, 5, 'over')
        loss_b = self.getClassifierLoss(true_pair, aug_pair, 20, 5, 'true')
        models['true_logit'] = [tf.argmax(self.getClassifierLogits(true_pair, 20, 'true' + str(i + 1)), axis=1) for i in range(5)]
        models['fake_logit'] = [tf.argmax(self.getClassifierLogits(aug_pair, 20, 'true' + str(i + 1)), axis=1) for i in range(5)]
        models['small_true_logit'] = [tf.argmax(self.getClassifierLogits(small_true_pair, 20, 'over' + str(i + 1)), axis=1) for i in range(5)]
        models['small_fake_logit'] = [tf.argmax(self.getClassifierLogits(small_aug_pair, 20, 'over' + str(i + 1)), axis=1) for i in range(5)]
        models['losses'] = [loss_a, loss_b]
        graph['over_ops'], graph['true_ops'] = self.getOptimizer(loss_a), self.getOptimizer(loss_b)
        performanceTracker(start, 'Graph completed in %fs')
        return graph, models, epoch_feed
