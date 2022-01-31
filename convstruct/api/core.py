import random
import os
import numpy as np
import tensorflow as tf
from itertools import chain
from random import randint
from PIL import Image, ImageDraw, ImageFont


class Core:
    def __init__(self, struct, gpu, memory, location, name):
        """
        :param struct: dict, dictionary of topologies to be used to create a graph.
        :param gpu: int, the gpu index.
        :param memory: int, amount of gpu memory to use, in gigabytes.
        :param location: str, directory name.
        :param name: str, string matching that of the name used for the created topology.
        """
        self.struct = struct
        self.gpu = gpu
        self.location = location
        self.name = name
        self.max_kernel_size = 9
        self.total_filters = 96 * memory
        self.max_graph_size = memory
        self.max_filter_size = 32 * memory

    def getTopology(self, batch_size, input_size, output_size, output_channels, input_channels, flat):
        """
        :param batch_size: int, the batch size to be fed to the graph.
        :param input_size: int, largest dimension of input between height and width.
        :param output_size: int, largest dimension of output between height and width.
        :param output_channels: int, number of channels to be returned from the graph.
        :param input_channels: int, number of channels to be fed to the graph.
        :param flat: bool, controls memory calculation for flat dims (flat / not flat example: 10, 1 / 10, 10).
        :return: This function returns the generated topology array and forms the topology parameters.
        """
        scaling = [calculateScale(input_size, output_size, self.max_graph_size) if self.struct['%s_classes' % self.name] == 0 else [2, 0], input_size, output_size, output_channels, input_channels, batch_size, flat]
        struct = self.getParameters(scaling)
        parameters = np.split(struct['%s_parameters_%d' % (self.name, self.gpu)], (len(self.struct['%s_split_%d' % (self.name, self.gpu)]) * 3) + (int(self.struct['%s_size_%d' % (self.name, self.gpu)]) * 2) + 5, 1)
        struct = self.setTopology(scaling, parameters)
        return struct

    def getParameters(self, scaling):
        """
        :param scaling: list, [scale, input_size, output_size, output_channels, batch_size].
        :return: This function returns an array of parameters to be fed to setTopology.
        """
        np.random.seed()
        self.struct['%s_size_%d' % (self.name, self.gpu)] = np.random.uniform(scaling[0][0], self.max_graph_size, [1])
        self.struct['%s_split_%d' % (self.name, self.gpu)] = splitRandom(int(self.struct['%s_size_%d' % (self.name, self.gpu)]), int(np.random.uniform(scaling[0][0], int(self.struct['%s_size_%d' % (self.name, self.gpu)]), [1])[0]))
        parameters = (len(self.struct['%s_split_%d' % (self.name, self.gpu)]) * 3) + (int(self.struct['%s_size_%d' % (self.name, self.gpu)]) * 2) + 5
        self.struct['%s_parameters_%d' % (self.name, self.gpu)] = np.reshape([random.uniform(0.1, 1.0) for _ in range(parameters)], (1, parameters))
        parameters = (list(chain.from_iterable(self.struct['%s_parameters_%d' % (self.name, self.gpu)].tolist())) + [0] * ((self.max_graph_size * 5) + 5))[:((self.max_graph_size * 5) + 5)]
        split = (self.struct['%s_split_%d' % (self.name, self.gpu)] + [0] * ((self.max_graph_size * 5) + 5))[:((self.max_graph_size * 5) + 5)]
        self.struct['%s_feed_%d' % (self.name, self.gpu)] = split + parameters + [self.struct['topology_index']]
        return self.struct

    def setGraph(self, graph_input, input_shape, train_ph, split):
        """
        :param graph_input: tensor, tensor batch of inputs to be fed to the graph.
        :param input_shape: list, shape of inputs to be fed to the graph using a 4d-tensor [batch size, width, height, channels].
        :param train_ph: tf.placeholder, tf.bool, controls batch normalization.
        :param split: numpy array, the number of parallel convolutions in a layer.
        :return: This function returns the graph using the topology parameters.
        """
        with tf.variable_scope(self.name if self.gpu == 0 else (self.name + '_' + str(self.gpu - 1)), reuse=tf.AUTO_REUSE):
            graph_input.set_shape(input_shape)
            graph_functions, layers = [tf.layers.conv2d, tf.layers.conv2d_transpose, tf.nn.leaky_relu, tf.nn.relu, tf.nn.tanh], dict()
            for i in range(len(split)):
                combine = []
                for ii in range(split[i]):
                    layers['input_%d_%d' % (ii, i)] = graph_functions[self.struct['%s_conv_%d_%d_%d' % (self.name, ii, i, self.gpu)]](
                        inputs=layers['input_0_%d' % (i - 1)] if i != 0 else graph_input,
                        filters=self.struct['%s_filter_%d_%d_%d' % (self.name, ii, i, self.gpu)],
                        kernel_size=self.struct['%s_kernel_%d_%d_%d' % (self.name, ii, i, self.gpu)],
                        strides=self.struct['%s_stride_%d_%d_%d' % (self.name, ii, i, self.gpu)],
                        padding="same",
                        name='input_%d_%d' % (ii, i))
                    layers['input_%d_%d' % (ii, i)] = tf.layers.batch_normalization(layers['input_%d_%d' % (ii, i)], training=train_ph)
                    layers['input_%d_%d' % (ii, i)] = graph_functions[self.struct['%s_activation_%d' % (self.name, self.gpu)]](layers['input_%d_%d' % (ii, i)])
                    if split[i] > 1:
                        combine.append(layers['input_%d_%d' % (ii, i)])
                        if split[i] == ii + 1 and len(split) != i + 1:
                            layers['input_0_%d' % i] = tf.concat(combine, 3)
                        elif split[i] == ii + 1 and len(split) == i + 1:
                            graph_input = tf.concat(combine, 3)
                    else:
                        graph_input = layers['input_%d_%d' % (ii, i)]
                if self.struct['%s_skip_%d_%d' % (self.name, i, self.gpu)] != 0 and len(split) != i + 1 and i != 0:
                    layers['input_0_%d' % i] = tf.concat([layers['input_0_%d' % i], layers['input_0_%d' % (self.struct['%s_skip_%d_%d' % (self.name, i, self.gpu)] - 1)]], 3)
            graph_input = graph_functions[self.struct['%s_conv_%d' % (self.name, self.gpu)]](graph_input, filters=self.struct['%s_filter_%d' % (self.name, self.gpu)], kernel_size=self.struct['%s_kernel_%d' % (self.name, self.gpu)], strides=1, padding="same")
            graph_input = tf.layers.batch_normalization(graph_input, training=train_ph) if self.struct['%s_norm_%d' % (self.name, self.gpu)] == 1 else graph_input
            graph_input = graph_functions[self.struct['%s_end_activation_%d' % (self.name, self.gpu)]](graph_input)
            return graph_input if self.struct['%s_classes' % self.name] == 0 else tf.layers.dense(tf.layers.flatten(graph_input), self.struct['%s_classes' % self.name], name='logits')

    def setTopology(self, scaling, parameters):
        """
        :param scaling: list, [scale, input_size, output_size, output_channels, input_channels, batch_size].
        :param parameters: array, an array of random float values between 0 and 1.
        :return: This function returns a dictionary of topology parameters to be fed to setGraph.
        """
        image = self.drawTopology(50, 50, '%s topology' % self.name, 4)
        image = self.drawTopology(50, 100, 'Input: ' + str((scaling[5], scaling[1], (1 if scaling[6] else scaling[1]), scaling[4])), 0, image)
        self.struct['%s_memory_%d' % (self.name, self.gpu)], combined_filters, counter, skip_min, skip_max, saved_concat = 0, 0, 0, 0, 0, scaling[4]
        split, total_filters = self.struct['%s_split_%d' % (self.name, self.gpu)], self.total_filters
        self.struct['%s_activation_%d' % (self.name, self.gpu)] = round((parameters[counter])[0].item()*2) + 2
        for i in range(len(split)):
            force_scale = True if scaling[0][0] == (len(split) - i) != 0 and self.struct['%s_classes' % self.name] == 0 else False
            force_reverse = True if scaling[0][0] < 0 and scaling[0][0] == -(len(split) - i) and self.struct['%s_classes' % self.name] == 0 else False
            force_stop = True if ((scaling[0][0] == 0 and len(split) == i + 1) or ((scaling[0][0] + 1 == (len(split) - i)) or (scaling[0][0] - 1 == -(len(split) - i)))) else False
            scaling[1] = ((scaling[1] * 2) if scaling[0][1] == 1 else (scaling[1] // 2)) if force_scale and not force_stop else scaling[1]
            scaling[1] = ((scaling[1] * 2) if scaling[0][1] == 0 else (scaling[1] // 2)) if force_reverse and not force_stop else scaling[1]
            scaling[1] = ((scaling[1] * 2) if round((parameters[counter + 1])[0].item()) == 1 else (scaling[1] // 2)) if not force_scale and not force_reverse and not force_stop and round((parameters[counter + 2])[0].item()) + 1 == 2 else scaling[1]
            scaling[0][0] -= 1 if (force_scale or (round((parameters[counter + 1])[0].item()) == scaling[0][1] and round((parameters[counter + 2])[0].item()) + 1 == 2)) and not force_stop and not force_reverse else 0
            scaling[0][0] += 1 if (force_reverse or (round((parameters[counter + 1])[0].item()) != scaling[0][1] and round((parameters[counter + 2])[0].item()) + 1 == 2)) and not force_stop and not force_scale else 0
            split_conv, split_stride = round((parameters[counter + 1])[0].item()), round((parameters[counter + 2])[0].item()) + 1
            skip_min = (i + 1) if round((parameters[counter + 2])[0].item()) + 1 == 2 or force_scale or force_reverse else skip_min
            skip_max += -skip_max if round((parameters[counter + 2])[0].item()) + 1 == 2 or force_scale or force_reverse else 1
            self.struct['%s_skip_%d_%d' % (self.name, i, self.gpu)] = (round((parameters[counter + 2])[0].item() * skip_max) + skip_min) * (round((parameters[counter + 2])[0].item()) if round((parameters[counter + 2])[0].item()) + 1 != 2 or not force_scale or not force_reverse and i != 0 else 0)
            saved_concat, total_filters, combined_filters, image = self.setTopologyLayer(parameters, i, split, split_conv, split_stride, total_filters, combined_filters, saved_concat, scaling, counter, [force_scale, force_reverse, force_stop], image)
            image = self.drawTopology(50, 50 * (i + 3 + (i * 2)) + 100, ('Concatenate Skip Connection: Layer %d' % (self.struct['%s_skip_%d_%d' % (self.name, i, self.gpu)] - 1)) if (self.struct['%s_skip_%d_%d' % (self.name, i, self.gpu)] != 0 and i != 0) else ('Concatenation' if split[i] > 1 else 'No Concatenation'), 3, image)
            counter += (split[i] * 2) + 3
        self.struct['%s_filter_%d' % (self.name, self.gpu)] = scaling[3]
        self.struct['%s_kernel_%d' % (self.name, self.gpu)] = round((parameters[counter])[0].item() * self.max_kernel_size) + 1
        self.struct['%s_conv_%d' % (self.name, self.gpu)] = round((parameters[counter + 1])[0].item())
        self.struct['%s_norm_%d' % (self.name, self.gpu)] = round((parameters[counter + 2])[0].item())
        self.struct['%s_end_activation_%d' % (self.name, self.gpu)] = round((parameters[counter + 3])[0].item()*2) + 2
        layer_parameters = ("Filters: " + str(self.struct['%s_filter_%d' % (self.name, self.gpu)]), " Kernel: " + str(self.struct['%s_kernel_%d' % (self.name, self.gpu)]), " Stride: 1")
        image = self.drawTopology(50, 50 * (len(split) + 1 + (len(split) * 2)) + 100, ('Convolution: ' if self.struct['%s_conv_%d' % (self.name, self.gpu)] == 0 else 'Deconvolution: ') + str(layer_parameters), 1 if len(split) % 2 == 0 else 2, image)
        activations = ['LeakyRelu', 'Relu', 'Tanh']
        self.struct['%s_output_node_%d' % (self.name, self.gpu)] = activations[self.struct['%s_end_activation_%d' % (self.name, self.gpu)] - 2]
        image = self.drawTopology(50, 50 * (len(split) + 2 + (len(split) * 2)) + 100, ((self.struct['%s_output_node_%d' % (self.name, self.gpu)] + ' Activation') + (' + Batch Normalization' if self.struct['%s_norm_%d' % (self.name, self.gpu)] == 1 else '') + ('' if self.struct['%s_classes' % self.name] == 0 else (' + Dense ' + str(self.struct['%s_classes' % self.name])))), 1 if len(split) % 2 == 0 else 2, image)
        self.drawTopology(50, 50 * (len(split) + 3 + (len(split) * 2)) + 100, 'Output: ' + (str((scaling[5], scaling[2], scaling[2], scaling[3])) if self.struct['%s_classes' % self.name] == 0 else str((scaling[5], scaling[3]))), 0, image)
        self.struct['%s_memory_%d' % (self.name, self.gpu)] += calculateMemory(self.struct['%s_memory_%d' % (self.name, self.gpu)], scaling[1], self.struct['%s_kernel_%d' % (self.name, self.gpu)], self.struct['%s_filter_%d' % (self.name, self.gpu)], saved_concat, scaling[5], scaling[6], True if self.struct['%s_classes' % self.name] > 0 else False)
        return self.struct

    def setTopologyLayer(self, parameters, i, split, split_conv, split_stride, total_filters, combined_filters, saved_concat, scaling, counter, forces, image):
        """
        :param parameters: array, an array of random float values between 0 and 1.
        :param i: int, layer index.
        :param split: list, list of individual and possible parallel layers at each layer.
        :param split_conv: int, value between 0 and 1 controlling whether layer uses a convolution or de-convolution.
        :param split_stride: int, value between 1 and 2 controlling layers stride.
        :param total_filters: int, total amount of filters that can be used in topology amongst all layers.
        :param combined_filters: int, total filters amongst parallel layers that will be concatenated together.
        :param saved_concat: int, total filters concatenated together in previous layer or input channels.
        :param scaling: list, [scale, input_size, output_size, output_channels, batch_size].
        :param counter: int, parameter index
        :param forces: list, [force_scale, force_reverse, force_stop]
        :param image: PIL.Image, topology graph image.
        :return: This function returns the indexed layer's parameters and adjusted topology parameters.
        """
        for ii in range(split[i]):
            remaining_layer_filters = self.max_filter_size if total_filters > self.max_filter_size else total_filters
            max_layer_filters = remaining_layer_filters if remaining_layer_filters > combined_filters else 8
            self.struct['%s_filter_%d_%d_%d' % (self.name, ii, i, self.gpu)] = round(parameters[(counter + 3) + (ii * 2)][0].item() * max_layer_filters) + 1
            self.struct['%s_kernel_%d_%d_%d' % (self.name, ii, i, self.gpu)] = round(parameters[(counter + 4) + (ii * 2)][0].item() * self.max_kernel_size) + 1
            self.struct['%s_stride_%d_%d_%d' % (self.name, ii, i, self.gpu)] = (2 if forces[0] or forces[1] else split_stride) if not forces[2] else 1
            self.struct['%s_conv_%d_%d_%d' % (self.name, ii, i, self.gpu)] = (scaling[0][1] if forces[0] else split_conv) if not forces[1] else scaling[0][2]
            total_filters -= self.struct['%s_filter_%d_%d_%d' % (self.name, ii, i, self.gpu)] if (total_filters - self.struct['%s_filter_%d_%d_%d' % (self.name, ii, i, self.gpu)]) >= 8 else 8
            combined_filters += -combined_filters if ii == 0 else self.struct['%s_filter_%d_%d_%d' % (self.name, ii, i, self.gpu)]
            self.struct['%s_memory_%d' % (self.name, self.gpu)] += calculateMemory(self.struct['%s_memory_%d' % (self.name, self.gpu)], scaling[1], self.struct['%s_kernel_%d_%d_%d' % (self.name, ii, i, self.gpu)], self.struct['%s_filter_%d_%d_%d' % (self.name, ii, i, self.gpu)], saved_concat, scaling[5], scaling[6])
            saved_concat = combined_filters if split[i] == ii + 1 else saved_concat
            layer_parameters = ("Filters: " + str(self.struct['%s_filter_%d_%d_%d' % (self.name, ii, i, self.gpu)]), " Kernel: " + str(self.struct['%s_kernel_%d_%d_%d' % (self.name, ii, i, self.gpu)]), " Stride: " + str(self.struct['%s_stride_%d_%d_%d' % (self.name, ii, i, self.gpu)]))
            image = self.drawTopology((410 * ii) + 50, 50 * (i + 1 + (i * 2)) + 100, ('Convolution: ' if self.struct['%s_conv_%d_%d_%d' % (self.name, ii, i, self.gpu)] == 0 else 'Deconvolution: ') + str(layer_parameters), 1 if i % 2 == 0 else 2, image)
            activations = ['LeakyRelu', 'Relu', 'Tanh']
            image = self.drawTopology((410 * ii) + 50, 50 * (i + 2 + (i * 2)) + 100, activations[self.struct['%s_activation_%d' % (self.name, self.gpu)] - 2] + ' Activation' + ' + Batch Normalization', 1 if i % 2 == 0 else 2, image)
        return saved_concat, total_filters, combined_filters, image

    def drawTopology(self, x_axis, y_axis, text, block, image=None):
        """
        :param x_axis: int, the width of the graph, taking into account parallel layers.
        :param y_axis: int, the height of the graph, taking into account number of sequential layers.
        :param text: str, block text.
        :param block: int, [input block, conv block 1, conv block 2, concat block].
        :param image: PIL.Image, previously created image to continue to add to.
        :return: This function returns a drawn topology and saves it as an image
        """
        if image is None:
            draw_width = max(self.struct['%s_split_%d' % (self.name, self.gpu)]) * 410 + 100
            draw_height = (len(self.struct['%s_split_%d' % (self.name, self.gpu)]) + 2) * 50 * 3 + 20
            image = Image.new('RGB', (draw_width, draw_height), color="#FFFFFF")
        blocks = [[(200, 40), "#5977DD"], [(400, 40), "#FF3B9D"], [(400, 40), "#C92075"], [(max(self.struct['%s_split_%d' % (self.name, self.gpu)]) * 406, 40), "#843060"], [(max(self.struct['%s_split_%d' % (self.name, self.gpu)]) * 406, 40), "#FFFFFF"]]
        new_block = ImageDraw.Draw(image)
        new_block.rectangle([(x_axis, y_axis), (x_axis + blocks[block][0][0], y_axis + blocks[block][0][1])], fill=blocks[block][1])
        font = ImageFont.truetype("arial", 30 if block == 4 else 12)
        text_size = font.getsize(text)
        text_size = (((x_axis + blocks[block][0][0] + x_axis) - text_size[0]) / 2, ((y_axis + blocks[block][0][1] + y_axis) - text_size[1]) / 2) if block != 4 else (50, 40)
        new_block.text(text_size, text, font=font, fill="#FFFFFF" if block != 4 else "#000000")
        if os.path.exists(os.path.join(self.struct['location'], os.path.join(self.struct['name'], '%s_topology_%d.jpg' % (self.name, self.gpu)))):
            os.chmod(os.path.join(self.struct['location'], os.path.join(self.struct['name'], '%s_topology_%d.jpg' % (self.name, self.gpu))), 0o777)
            os.remove(os.path.join(self.struct['location'], os.path.join(self.struct['name'], '%s_topology_%d.jpg' % (self.name, self.gpu))))
        image.save(os.path.join(self.struct['location'], os.path.join(self.struct['name'], '%s_topology_%d.jpg' % (self.name, self.gpu))), "JPEG")
        return image


def calculateMemory(current_memory, input_size, kernel, filters, saved_concat, batch_size, flat, dense=False):
    """
    :param current_memory: int, the current sum of memory.
    :param input_size: int, largest dimension of input between height and width.
    :param kernel: int, kernel size set for current layer.
    :param filters: int, filter size set for current layer.
    :param saved_concat: int, concatenated value of all parallel layer filters.
    :param batch_size: int, the batch size to be fed to the graph.
    :param flat: bool, controls memory calculation for flat dims (flat / not flat example: 10, 1 / 10, 10).
    :param dense: bool, controls memory calculation for dense op.
    :return: Returns the sum of memory used in a layer.
    """
    current_memory += kernel * kernel * saved_concat * filters * 3
    current_memory += filters * (input_size if flat else (input_size**2)) * batch_size * 2 * 2 * (filters if dense else 1)
    current_memory += saved_concat * (input_size if flat else (input_size**2)) * batch_size * 2
    return current_memory


def splitRandom(a, n):
    """
    :param a: int, main number.
    :param n: int, number to split main number by.
    :return: Returns simple split of a into n pieces.
    """
    pieces = []
    n = 1 if n == 0 else n
    if a == n:
        for _ in range(n):
            pieces.append(1)
    else:
        for idx in range(n - 1):
            pieces.append(randint(1, a - sum(pieces) - n + idx))
        pieces.append(a - sum(pieces))
    return pieces


def calculateScale(a, b, m):
    """
    :param a: int, starting value.
    :param b: int, end value.
    :param m: int, max graph size.
    :return: Returns list of scaling parameters.
    """
    scale = 0
    scaled_x = a if a > b else b
    scaled_y = b if a > b else a
    while True:
        if scaled_x != scaled_y and scaled_x != 1:
            scale += 1
            scaled_x /= 2
            scaled_x = 1 if scaled_x < 1 else scaled_x
        else:
            break
    if scale > m:
        exit(print("[Error]:Topology max memory set to", m, "GB. Minimum memory required for this topology is", scale, "GB."))
    return [scale, 0 if a > b else 1, 1 if a > b else 0]
