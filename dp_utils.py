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

"""Utils for building and training NN models.
"""
from __future__ import division

import math

import numpy
import tensorflow as tf


def BatchClipByL2norm(t, upper_bound, name=None):
  """Clip an array of tensors by L2 norm.

  Shrink each dimension-0 slice of tensor (for matrix it is each row) such
  that the l2 norm is at most upper_bound. Here we clip each row as it
  corresponds to each example in the batch.

  Args:
    t: the input tensor.
    upper_bound: the upperbound of the L2 norm.
    name: optional name.
  Returns:
    the clipped tensor.
  """

  assert upper_bound > 0
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t


def AddGaussianNoise(t, sigma, name=None):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """

  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t
