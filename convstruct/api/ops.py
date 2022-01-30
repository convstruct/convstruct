import tensorflow as tf


def softmaxLoss(loss, graph, gpu):
    """
    :param loss: string, the name used for the generated topology.
    :param graph: dict, dictionary of gpu batches of created graphs.
    :param gpu: int, the gpu index.
    :return: This function returns a softmax cross entropy loss.
    """
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=graph['real'], logits=graph['%s_output_%d' % (loss, gpu)])
    return tf.reduce_mean(entropy)


def sigmoidLoss(loss, graph, gpu):
    """
    :param loss: string, the name used for the generated topology.
    :param graph: dict, dictionary of gpu batches of created graphs.
    :param gpu: int, the gpu index.
    :return: This function returns a sigmoid cross entropy loss.
    """
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=graph['real'], logits=graph['%s_output_%d' % (loss, gpu)])
    return tf.reduce_mean(entropy)


def mseLoss(loss, graph, gpu):
    """
    :param loss: string, the name used for the generated topology.
    :param graph: dict, dictionary of gpu batches of created graphs.
    :param gpu: int, the gpu index.
    :return: This function returns a mean square error loss.
    """
    return tf.reduce_mean(tf.square(graph['real'] - graph['%s_output_%d' % (loss, gpu)]))


def L2Loss(loss, graph, gpu):
    """
    :param loss: string, the name used for the generated topology.
    :param graph: dict, dictionary of gpu batches of created graphs.
    :param gpu: int, the gpu index.
    :return: This function returns a squared L2 loss.
    """
    trainable_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * 0.01
    return l2_loss + tf.reduce_sum(tf.square(graph['real'] - graph['%s_output_%d' % (loss, gpu)]))


def gpLoss(loss, graph, gpu):
    """
    :param loss: list of strings, [generator output logits, real data logits, sum logits, sum data].
    :param graph: dict, dictionary of gpu batches of created graphs.
    :param gpu: int, the gpu index.
    :return: This function returns a gradient penalty loss.
    """

    if loss[2] is None:
        loss = -tf.reduce_mean(graph['%s_output_%d' % (loss[0], gpu)])
    else:
        gradient_penalty = tf.gradients(graph['%s_output_%d' % (loss[2], gpu)], [graph['%s_output_%d' % (loss[3], gpu)]])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient_penalty), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.) ** 2)
        gan = tf.reduce_mean(graph['%s_output_%d' % (loss[0], gpu)]) - tf.reduce_mean(graph['%s_output_%d' % (loss[1], gpu)])
        loss = gan + 10.0 * gradient_penalty
    return loss


def adamOp(loss, name, learning_rate=1e-4, beta1=0., beta2=0.9):
    """
    :param loss: tf.loss, the graph loss function.
    :param name: string, the name of the scope.
    :param learning_rate: int, learning rate for the adam optimizer parameter.
    :param beta1: int, beta1 for the adam optimizer parameter.
    :param beta2: int, beta2 for the adam optimizer parameter.
    :return: This function returns an adam optimizer.
    """
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
    with tf.control_dependencies(ops):
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(loss, var_list=var)


def moOp(loss, name, learning_rate=0.1, momentum=0.9, nesterov=True):
    """
    :param loss: tf.loss, the graph loss function.
    :param name: string, the name of the scope.
    :param learning_rate: float, learning rate for the momentum parameter.
    :param momentum: float, momentum for the momentum parameter.
    :param nesterov: bool, controls use of nesterov in momentum optimizer.
    :return: This function returns a momentum optimizer.
    """
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
    with tf.control_dependencies(ops):
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=nesterov).minimize(loss, var_list=var)


def RMSPropOp(loss, name, learning_rate=1e-3, decay=0.9, momentum=0.0, epsilon=1e-10):
    """
    :param loss: tf.loss, the graph loss function.
    :param name: string, the name of the scope.
    :param learning_rate: int, learning rate for the adam optimizer parameter.
    :param decay: float, discounting factor for the coming gradient.
    :param momentum: float, scalar tensor.
    :param epsilon: float, small value to avoid zero denominator.
    :return: This function returns an RMSProp optimizer.
    """
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
    with tf.control_dependencies(ops):
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=epsilon).minimize(loss, var_list=var)
