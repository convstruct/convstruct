import glob
import logging
import os
import random
import argparse
import time
import numpy as np
import tensorflow as tf
import subprocess as sp
from tensorflow.python.client import device_lib
from PIL import Image, ImageEnhance
from os import path
from random import randint

# ----- To repress Tensorflow deprecation warnings ----- #
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def args_parser():
    """
    :return: This function returns arguments to setup the Convstruct class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--compdir', type=str, help='Folder location for ground truth images.')
    parser.add_argument('--indir', type=str, help='Folder location for starting point images.')
    parser.add_argument('--num_comp', type=int, default=1, help='Number of ground truth images.')
    parser.add_argument('--num_in', type=int, default=1, help='Number of starting points.')
    parser.add_argument('--num_type', type=str, default='random', help='Type of data feeding: ordered or random.')
    parser.add_argument('--name', type=str, default='v1', help='Name of main folder.')
    parser.add_argument('--console', action='store_false', help='Print progress to console.')
    args = parser.parse_args()
    return args


def process_image(length, stage, feed, width, height):
    """
    :param length: feed length.
    :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
    :param feed: image/s location.
    :param width: width of image.
    :param height: height of image.
    :return: This function returns the processed image data and resizes if required.
    """
    image_string = tf.read_file(feed)
    image_decoded = tf.io.decode_image(image_string, 3, expand_animations=False)
    image_decoded = tf.image.resize(image_decoded, [height, width])
    if stage != 1 and length < 50000:
        image_decoded = tf.image.random_brightness(image_decoded, max_delta=0.1)
        image_decoded = tf.image.random_contrast(image_decoded, lower=0.9, upper=1.1)
        image_decoded = tf.image.random_flip_left_right(image_decoded)
    image_decoded.set_shape((height, width, 3))
    return image_decoded


def build_input_pipeline(args, filenames, growth, random_seed, input_type, num_input, stage, shuffle=True, num_threads=4):
    """
    :param args: dictionary of arguments.
    :param filenames: list of file names.
    :param growth: dictionary of info on input and output data.
    :param random_seed: random value to feed np.random.seed.
    :param input_type: the folder location of ground truth images or starting points.
    :param num_input: the number of ground truth or starting points being fed.
    :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
    :param shuffle: randomize batches.
    :param num_threads: tf.train.batch threads to use.
    :return: This function returns a tensor batch queue from files provided.
    """
    start = time.time()
    np.random.seed(random_seed)
    train_list, image_batch = [] if args['num_type'] == 'ordered' else tf.constant(filenames), []
    if args['num_type'] == 'ordered':
        for i in range(num_input):
            files = tf.constant(filenames) if i == 0 else tf.constant(filenames[((i * num_input)-1):])
            train_list.append(files)
    for i in range(num_input):
        filename_queue = tf.train.string_input_producer(train_list if args['num_type'] == 'random' else train_list[i], shuffle=shuffle if args['num_type'] == 'random' else False)
        image = process_image(len(filenames), stage, filename_queue.dequeue(), growth['x_size'] if stage != 2 or args['indir'] else growth['small_x_size'], growth['y_size'] if stage != 2 or args['indir'] else growth['small_y_size'])
        image_batched = tf.train.batch([image], batch_size=16 if stage == 1 else growth["batch_size"], num_threads=num_threads, capacity=10 * (16 if stage == 1 else growth["batch_size"]))
        image_batch.append(image_batched)
    performance_tracker(start, 'Ground truth pipeline completed in %fs' if input_type == args['compdir'] else 'Input pipeline completed in %fs')
    return image_batch if num_input > 1 else image_batch[0]


def build_inputs(args, input_type, num_input, growth, stage):
    """
    :param args: dictionary of arguments.
    :param input_type: the folder location of ground truth images or starting points.
    :param num_input: the number of ground truth or starting points being fed.
    :param growth: dictionary of info on input and output data.
    :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
    :return: This function sets a seed to randomly select filenames and sends filenames to be built into tensors.
    """
    start = time.time()
    random_seed = random.randrange(1000)
    feed = np.array(glob.glob(os.path.join(input_type, '**', '*.*'), recursive=True))
    get_batch = build_input_pipeline(args, feed, growth, random_seed, input_type, num_input, stage)
    performance_tracker(start, 'Build inputs completed in %fs')
    return get_batch


def setup_batch(args, sess, get_ground_truth_batch, get_starting_points_batch, growth=None):
    """
    :param args: dictionary of arguments.
    :param sess: tensorflow active session.
    :param get_ground_truth_batch: create batch of ground truth images. 
    :param get_starting_points_batch: create batch of starting point images.
    :param growth: dictionary of info on input and output data.
    :return: This function returns preprocessed batches for the current training loop.
    """
    epochs_starting_points, epochs_ground_truth, epochs_augmented = dict(), dict(), dict()
    start = time.time()
    ground_truth_batch = sess.run(get_ground_truth_batch)
    performance_tracker(start, 'ground truth batch completed in %fs')
    if args['indir'] and growth is not None:
        start = time.time()
        starting_points_batch = sess.run(get_starting_points_batch)
        performance_tracker(start, 'starting point batch completed in %fs')
        for i in range(args['num_in']):
            epochs_starting_points['%d' % i] = starting_points_batch[i] if args['num_in'] > 1 else starting_points_batch
    elif growth is not None:
        for i in range(args['num_in']):
            epochs_starting_points['%d' % i] = np.random.rand(growth["batch_size"], 100).astype(np.float32) * 255.0
    for i in range(args['num_comp']):
        epochs_ground_truth['%d' % i] = ground_truth_batch[i] if args['num_comp'] > 1 else ground_truth_batch
        if growth is None:
            augmented_batch = augment_batch(ground_truth_batch[i] if args['num_comp'] > 1 else ground_truth_batch)
            epochs_augmented['%d' % i] = augmented_batch
    epochs_ground_truth = epochs_ground_truth if args['num_comp'] > 1 else epochs_ground_truth['0']
    epochs_starting_points = (epochs_starting_points if args['num_in'] > 1 else epochs_starting_points['0']) if growth is not None else 0
    epochs_augmented = (epochs_augmented if len(epochs_augmented) > 1 else epochs_augmented['0']) if growth is None else 0
    return epochs_ground_truth, epochs_starting_points, epochs_augmented


def setup_mini_batch(feed, specifications):
    """
    :param feed: module and model parameter feed.
    :param specifications: dictionary of convstruct settings and system details.
    :return: This function returns a mini batch of module quality scores and their corresponding parameters.
    """
    random.shuffle(feed)
    quality = []
    teachers = []
    for i in range(16):
        quality.append(feed[i][0])
        teacher_normalised = (feed[i][1] - np.min(feed[i][1])) / (np.max(feed[i][1]) - np.min(feed[i][1]))
        teachers.append(teacher_normalised)
    quality_batch = np.reshape(np.array([quality]), (16, 1))
    teacher_batch = np.reshape(np.array([teachers]), (16, (specifications['max_module_size'] * 2) + 1))
    return quality_batch, teacher_batch


def build_log_dir(main_name, path_name=None, second_path_name=None):
    """
    :param main_name: top folder name.
    :param path_name: child folder name.
    :param second_path_name: child within a child folder name.
    :return: Returns path to main folder and creates the folder.
    """
    log_path = main_name
    if second_path_name is not None:
        log_path = os.path.join(log_path, path_name)
        log_path = os.path.join(log_path, second_path_name)
    elif path_name is not None:
        log_path = os.path.join(log_path, path_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path


def create_pair(image):
    """
    :param image: image data to create a pair from.
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


def augment_batch(batch):
    """
    :param batch: batch of images.
    :return: This function returns a batch of augmented pairs.
    """
    start = time.time()
    augmented_pair = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 3))
    for i in range(batch.shape[0]):
        augmented_pair[i, :, :, :] = create_pair(batch[i, :, :, :])
    performance_tracker(start, 'Augment batch completed in %fs')
    return augmented_pair


def convstruct_start(args, location):
    """
    :param args: dictionary of arguments.
    :param location: path of directory to save weights and summaries to.
    :return: Creates/loads specifications dictionary and growth dictionary.
    """
    start = time.time()
    feed_comp = np.array(glob.glob(os.path.join(args['compdir'], '**', '*.*'), recursive=True))
    specifications = np.load(os.path.join(location, 'specifications.npy')).flat[0] if path.exists(os.path.join(location, 'specifications.npy')) else {'max_memory': 0, 'max_kernel_size': 9, 'max_stride_size': 2, 'total_epoch_count': 100, 'epoch_count': 0, 'previous_progress': 0, 'live_learning': True, 'multi_gpu_test': False}
    if tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None):
        local_devices = device_lib.list_local_devices()
        gpus = [x.name for x in local_devices if x.device_type == 'GPU']
        for gpu in range(len(gpus)):
            if gpu == 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] += ',%d' % gpu
            specifications['max_memory_%d' % gpu], specifications['max_memory'] = get_gpu_memory(gpu), get_gpu_memory(gpu) if (specifications['max_memory'] > get_gpu_memory(gpu) or gpu == 0) else specifications['max_memory']
        local_devices = device_lib.list_local_devices()
        gpus = [x.name for x in local_devices if x.device_type == 'GPU']
        specifications['gpus'] = len(gpus)
    else:
        specifications['gpus'] = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        specifications['max_memory_0'], specifications['max_memory'] = 6000000000, 6000000000
    specifications['total_filters'], specifications['max_module_size'], specifications['max_filter_size'] = ((2048, 45, 512) if specifications['max_memory'] > 17000000000 else (1536, 36, 512)) if specifications['max_memory'] > 7000000000 else (1024, 27, 512)
    if path.exists(os.path.join(os.path.join(location, 'draw'), 'growth.npy')):
        growth = np.load(os.path.join(os.path.join(location, 'draw'), 'growth.npy')).flat[0]
    elif path.exists(os.path.join(location, 'growth.npy')):
        growth = np.load(os.path.join(location, 'growth.npy')).flat[0]
    else:
        growth = {'model_target': 0, 'tested': False, 'teacher_feeds': [], 'model_quality': [], 'draw_learning': True, 'saved_epoch': 0, 'initial_length': 10000, 'full_length': 100000, 'iicc_length': 1000, 'estimator_length': 20000, 'batch_size': 32}
        comp_sample = randint(0, len(feed_comp) - 1)
        image = Image.open(feed_comp[comp_sample])
        image_x, image_y = image.size[0], image.size[1]
        smaller_dim = image_x if image_x < image_y else image_y
        sizes = (256 if specifications['max_memory'] > 17000000000 else 128) if specifications['max_memory'] > 7000000000 else 64
        growth['x_size'], growth['y_size'] = (sizes if smaller_dim >= sizes else 128) if (smaller_dim >= 128 and sizes >= 128) else 64, (sizes if smaller_dim >= sizes else 128) if (smaller_dim >= 128 and sizes >= 128) else 64
        growth['small_x_size'], growth['small_y_size'] = growth['x_size'] // 2 // 2, growth['y_size'] // 2 // 2
        growth['scaling'] = (5 if growth['x_size'] == 256 else (4 if growth['x_size'] == 128 else 3)) if not args['indir'] else 2
        image.close()
    performance_tracker(start, 'Start preparations completed in %fs')
    return specifications, growth


def split_random(a, n):
    """
    :param a: main number.
    :param n: number to split main number by.
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


def multi_placeholders(count, ph_name, noise=1):
    """
    :param count: number of placeholders to return.
    :param ph_name: name of placeholder.
    :param noise: indicator value of placeholder shape.
    :return: Returns a single or multiple placeholders.
    """
    placeholder = dict()
    for i in range(count):
        placeholder['%d' % i] = tf.placeholder(tf.float32, [None, None] if noise is None else [None, None, None, 3], name=('%s_ph_%d' % (ph_name, i)) if count > 1 else ('%s_ph' % ph_name))
    return placeholder if count > 1 else placeholder['0']


def multi_dict_feed(args, dict_feed, in_ph, in_feed, comp_ph, comp_feed, stage):
    """
    :param args: dictionary of arguments.
    :param dict_feed: epoch feed being returned.
    :param in_ph: starting point placeholder.
    :param in_feed: starting point batch feed.
    :param comp_ph: ground truth placeholder.
    :param comp_feed: ground truth batch feed.
    :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
    :return: Returns a single or multiple feeds of batches.
    """
    if args['num_comp'] > 1:
        for i, value in comp_feed.items():
            dict_feed[comp_ph['%d' % int(i)]] = value
    else:
        dict_feed[comp_ph] = comp_feed
    if ((stage == 1 or stage == '2_end') and args['num_comp'] > 1) or (stage != 1 and stage != '2_end' and args['num_in'] > 1):
        if stage == '2_end':
            for i in range(args['num_comp']):
                dict_feed[in_ph['%d' % int(i)]] = in_feed[i]
        else:
            for i, value in in_feed.items():
                dict_feed[in_ph['%d' % int(i)]] = value
    else:
        dict_feed[in_ph] = in_feed
    return dict_feed


def calculate_target(qualities):
    """
    :param qualities: module quality scores.
    :return: Returns target quality score.
    """
    filtered = qualities.copy()
    mi = min(filtered)
    b = [mi] + [x for x in filtered if x != mi]
    target = int(sum(b) / len(b))
    return target


def performance_tracker(start, string, index=None):
    """
    :param start: time.time() start variable.
    :param string: message to be used in logging.
    :param index: secondary formatting.
    :return: Returns calculated performance and prints to log if above threshold.
    """
    end = time.time()
    if (end - start) > 10:
        logging.debug(string % (index, end - start) if index is not None else string % (end - start))
    return


def tf_config_setup():
    """
    :return: Returns tensorflow configuration.
    """
    tf_config = tf.ConfigProto(allow_soft_placement=False)
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.polling_inactive_delay_msecs = 50
    return tf_config


def create_log(location, string):
    """
    :param location: location to create log in.
    :param string: log name.
    :return: Returns
    """
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    log_location = os.path.join(location, string)
    logging.basicConfig(filename=log_location, level=logging.DEBUG)
    return


def get_gpu_memory(gpu):
    """
    :param gpu: the gpu index.
    :return: Returns the total memory of the indexed Nvidia GPU using nvidia-smi.
    """
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    proc = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, stdin=sp.PIPE)
    proc.stdin.close()
    proc.wait()
    memory_total_info = proc.stdout.read()
    memory_total_values = [int(s) for s in memory_total_info.split() if s.isdigit()]
    return memory_total_values[gpu] * 1000000


def initialization(sess):
    """
    :param sess: tensorflow active session.
    :return: Returns the initialized tensorflow session.
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    start = time.time()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    performance_tracker(start, 'Initialization completed in %fs')
    return coord, threads


def preprocess(args, starting, ground, augmented, stage):
    """
    :param args: dictionary of arguments.
    :param starting: starting point images.
    :param ground: ground truth images.
    :param augmented: augmented pairs of ground truth images.
    :param stage: value indicator to identify between learn(1), live(2), and draw(3/4).
    :return: This function returns the preprocessed image batches.
    """
    if stage != 1:
        if args['num_in'] > 1:
            for i, value in starting.items():
                starting[i] = (starting[i] / 127.5) - 1
        else:
            starting = (starting / 127.5) - 1
    if args['num_comp'] > 1:
        for i, value in ground.items():
            ground[i] = (ground[i] / 127.5) - 1
            if stage == 1:
                augmented[i] = (augmented[i] / 127.5) - 1
    else:
        ground = (ground / 127.5) - 1
        if stage == 1:
            augmented = (augmented / 127.5) - 1
    return starting, ground, augmented
