import gzip
import os

# GPUID = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

from scipy import ndimage
from six.moves import urllib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import sys
import scipy.io


import pdb

print ("PACKAGES LOADED")



def CNN(inputs, _is_training=True):
    x   = tf.reshape(inputs, [-1, 28, 28, 1])
    batch_norm_params = {'is_training': _is_training, 'decay': 0.9, 'updates_collections': None}
    net = slim.conv2d(x, 32, [5, 5], padding='SAME'
                     , activation_fn       = tf.nn.relu
                     , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                     , normalizer_fn       = slim.batch_norm
                     , normalizer_params   = batch_norm_params
                     , scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 1024
                    , activation_fn       = tf.nn.relu
                    , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                    , normalizer_fn       = slim.batch_norm
                    , normalizer_params   = batch_norm_params
                    , scope='fc4')
    net = slim.dropout(net, keep_prob=0.7, is_training=_is_training, scope='dropout4')
    out = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return out


# DATA URL
SOURCE_URL      = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY  = "data"
# PARAMETERS FOR MNIST
IMAGE_SIZE      = 28
NUM_CHANNELS    = 1
PIXEL_DEPTH     = 255
NUM_LABELS      = 10
VALIDATION_SIZE = 5000  # Size of the validation set.

# DOWNLOAD MNIST DATA, IF NECESSARY
def maybe_download(filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

# EXTRACT IMAGES
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH # -0.5~0.5
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data # [image index, y, x, channels]

# EXTRACT LABELS
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

# AUGMENT TRAINING DATA
def expend_training_data(images, labels):
    expanded_images = []
    expanded_labels = []
    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        # APPEND ORIGINAL DATA
        expanded_images.append(x)
        expanded_labels.append(y)
        # ASSUME MEDIAN COLOR TO BE BACKGROUND COLOR
        bg_value = np.median(x) # this is regarded as background's value
        image = np.reshape(x, (-1, 28))

        for i in range(4):
            # ROTATE IMAGE
            angle = np.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
            # SHIFT IAMGE
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)
            # ADD TO THE LIST
            expanded_images.append(np.reshape(new_img_, 784))
            expanded_labels.append(y)
    expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)
    np.random.shuffle(expanded_train_total_data)
    return expanded_train_total_data

# PREPARE MNIST DATA
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = np.concatenate((train_data, train_labels), axis=1)
    train_size = train_total_data.shape[0]
    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels



# CONFIGURATION
# MODEL_DIRECTORY   = "./model/model.ckpt"
MODEL_DIRECTORY   = "../model/model.ckpt"
LOGS_DIRECTORY    = "logs/train"
training_epochs   = 10
TRAIN_BATCH_SIZE  = 50
display_step      = 500
validation_step   = 500
TEST_BATCH_SIZE   = 5000


# PREPARE MNIST DATA
batch_size = TRAIN_BATCH_SIZE # BATCH SIZE (50)
num_labels = NUM_LABELS       # NUMBER OF LABELS (10)
# train_total_data, train_size, validation_data, validation_labels \
#     , test_data, test_labels = prepare_MNIST_data(False)
# PRINT FUNCTION
def print_np(x, str):
    print (" TYPE AND SHAPE OF [%18s ] ARE %s and %14s"
           % (str, type(x), x.shape,))
# print_np(train_total_data, 'train_total_data')
# print_np(validation_data, 'validation_data')
# print_np(validation_labels, 'validation_labels')
# print_np(test_data, 'test_data')
# print_np(test_labels, 'test_labels')
################################################################################



# DEFINE MODEL
# PLACEHOLDERS
x  = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) #answer
is_training = tf.placeholder(tf.bool, name='MODE')
# CONVOLUTIONAL NEURAL NETWORK MODEL
y = CNN(x, is_training)
# DEFINE LOSS
with tf.name_scope("LOSS"):
    loss = slim.losses.softmax_cross_entropy(y, y_)
# DEFINE ACCURACY
with tf.name_scope("ACC"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


# OPEN SESSION
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer(), feed_dict={is_training: False})


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(e_x.sum(axis=1), axis=1)   # only difference

# RESTORE SAVED NETWORK
saver.restore(sess, MODEL_DIRECTORY)
print("model restored")

from config import *
import joblib

args = parse_arguments()

# folders for generated images
result_folder = './ali_results/'

icp = []
try:
    data = joblib.load(args.exp_name)
except:
    from tqdm import tqdm

    data = np.zeros((100000, 794))
    dim = 0
    for i in tqdm(range(10)):
        data_x = joblib.load(args.exp_name + f'-{i}.pkl')
        data[dim: dim + len(data_x)] = data_x
        dim += len(data_x)
    print(data.shape)
if 'test' not in args.exp_name:
    test_x = data[:, :784]
    test_x -= 0.5

for k in range(3):
    k = k + 1
    # result_folder = '../GAN/ali_bigan/ali_shell_results/'
    # mat = scipy.io.loadmat(result_folder+ '2_2_2_256_256_1024.mat' )
    # mat = scipy.io.loadmat(result_folder+ '{}.mat'.format(str(k).zfill(3)))
    from sklearn.utils import shuffle
    test_x = shuffle(test_x)
    # label = label.reshape((label.shape[0], nb_classes), order='F')
    test_data = test_x.reshape(test_x.shape[0], 784)

    # test_data  = mat['images']
    #  pdb.set_trace()
    # COMPUTE ACCURACY FOR TEST DATA
    batch_size = 512
    test_size   = test_data.shape[0]
    total_batch = int(test_size / batch_size)
    acc_buffer  = []
    preds = []
    for i in range(total_batch):
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size), :]
        y_final = sess.run(y, feed_dict={x: batch_xs, is_training: False})
        pred_softmax = softmax(y_final)
        preds.append(pred_softmax)


    preds = np.concatenate(preds, 0)
    scores = []
    splits = 10
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))

    icp.append((np.mean(scores) , np.std(scores)))
    print("Inception score is: %.4f, %.4f" % (np.mean(scores) , np.std(scores)))

# scipy.io.savemat('ali_inception_50.mat', mdict={'icp': icp})
joblib.dump(icp, 'inception.pkl')
