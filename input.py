# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# TODO update load mnist

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _pickle as cPickle
import gzip
import math
import numpy as np
import os
from scipy.io import loadmat as loadmat
from six.moves import urllib
from six.moves import xrange
import sys
import tarfile

import tensorflow as tf

def create_dir_if_needed(dest_directory):
    """
    Create directory if doesn't exist
    :param dest_directory:
    :return: True if everything went well
    """
    if not tf.gfile.IsDirectory(dest_directory):
        tf.gfile.MakeDirs(dest_directory)

    return True

    # Test if file already exists
    if not tf.gfile.Exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(file_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    return result


def image_whitening(data):
    """
    Subtracts mean of image and divides by adjusted standard variance (for
    stability). Operations are per image but performed for the entire array.
    :param image: 4D array (ID, Height, Weight, Channel)
    :return: 4D array (ID, Height, Weight, Channel)
    """
    assert len(np.shape(data)) == 4

    # Compute number of pixels in image
    nb_pixels = np.shape(data)[1] * np.shape(data)[2] * np.shape(data)[3]

    # Subtract mean
    mean = np.mean(data, axis=(1,2,3))

    ones = np.ones(np.shape(data)[1:4], dtype=np.float32)
    for i in xrange(len(data)):
        data[i, :, :, :] -= mean[i] * ones

    # Compute adjusted standard variance
    adj_std_var = np.maximum(np.ones(len(data), dtype=np.float32) / math.sqrt(nb_pixels), np.std(data, axis=(1,2,3))) #NOLINT(long-line)

    # Divide image
    for i in xrange(len(data)):
        data[i, :, :, :] = data[i, :, :, :] / adj_std_var[i]

    print(np.shape(data))

    return data    

def ld_mnist(data_dir, dataset_name):
        
    data_dir = os.path.join(data_dir, dataset_name)
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)

    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec


def partition_dataset(data, labels, nb_teachers, teacher_id):
    """
    Simple partitioning algorithm that returns the right portion of the data
    needed by a given teacher out of a certain nb of teachers
    :param data: input data to be partitioned
    :param labels: output data to be partitioned
    :param nb_teachers: number of teachers in the ensemble (affects size of each
                       partition)
    :param teacher_id: id of partition to retrieve
    :return:
    """

    # Sanity check
    assert(len(data) == len(labels))
    assert(int(teacher_id) < int(nb_teachers))

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    # Compute start, end indices of partition
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len

    # Slice partition off
    partition_data = data[start:end]
    partition_labels = labels[start:end]

    return partition_data, partition_labels
