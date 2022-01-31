import glob
import logging
import os
import numpy as np
import tensorflow as tf
from PIL import Image


def processImage(feed, dims, augment=False):
    """
    :param feed: str, image/s location.
    :param dims: list, the b,w,h,c dimensions of the input data ([None, w, h, c]).
    :param augment: bool, boolean value controlling whether the image data is augmented.
    :return: This function returns the processed image data and resizes if required.
    """
    image_string = tf.read_file(feed)
    image_decoded = tf.io.decode_image(image_string, dims[2], expand_animations=False)
    image_decoded = tf.image.resize(image_decoded, [dims[1], dims[0]])
    if augment:
        image_decoded = tf.image.random_brightness(image_decoded, max_delta=0.1)
        image_decoded = tf.image.random_contrast(image_decoded, lower=0.9, upper=1.1)
        image_decoded = tf.image.random_flip_left_right(image_decoded)
    image_decoded.set_shape((dims[1], dims[0], dims[2]))
    return image_decoded


def createBatch(directory, dims, batch_size, augment=False, shuffle=True):
    """
    :param directory: str, the folder location of real data or input data.
    :param dims: list, the w,h,c dimensions of the input data ([None, w, h, c]).
    :param batch_size: int, size of batch to feed to a graph.
    :param augment: bool, boolean value controlling whether the image data is augmented.
    :param shuffle: bool, randomize batches.
    :return: This function returns a tensor batch using files provided.
    """
    feed = np.array(glob.glob(os.path.join(directory, '**', '*.*'), recursive=True))
    images = tf.convert_to_tensor(feed, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(lambda x: processImage(x, dims, augment))
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator


def createLog(location, string):
    """
    :param location: str, folder to create log in.
    :param string: str, log name.
    :return: Returns a created log.
    """
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    log_location = os.path.join(location, string)
    logging.basicConfig(filename=log_location, level=logging.DEBUG)
    return


def saveImage(images, name):
    """
    :param images: array float32, graph batch output images to be saved.
    :param name: string, name of topology used to generate image.
    :return: Returns saved images after being post processed.
    """
    for i in range(len(images)):
        Image.fromarray((((images[i] + 1) / 2) * 255).astype(np.uint8)).save(os.path.join(name, "%s_image_%d.jpg" % (name, i)))
    return


def maxDuration(action):
    """
    :param action: list, int representing an action, per gpu.
    :return: Returns the max episode duration size within all gpu sessions.
    """
    max_size = 0
    for gpu in range(len(action)):
        duration = [10, 100, 200, 0][action[gpu]]
        if duration > max_size:
            max_size = duration
    return max_size
