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

""" Combined sanitizer.py and dp_pca.py under tensorflow/models/research/differential-privacy """

"""Differentially private optimizers.
"""
import tensorflow as tf
import collections
from sklearn.preprocessing import normalize
from rdp_utils import gaussian_rdp
import numpy as np

def ComputeDPPrincipalProjection(data, projection_dims, orders, sigma):
  """Compute differentially private projection.

  Args:
    data: the input data, each row is a data vector.
    projection_dims: the projection dimension.
    sigma: sigma for gaussian noise
  Returns:
    A projection matrix with projection_dims columns.
  """

  # Normalize each row.
  normalized_data = normalize(data, norm='l2', axis=1)
  covar = np.matmul(np.transpose(normalized_data), normalized_data)

  # Since the data is already normalized, there is no need to clip
  # the covariance matrix.

  gaussian_noise, rdp_budget = gaussian_rdp(covar.reshape([1,-1]), 1.0, orders, sigma)

  saned_covar = covar + gaussian_noise.reshape(covar.shape)

  # Symmetrize saned_covar. This also reduces the noise variance.
  saned_covar = 0.5 * (saned_covar + np.transpose(saned_covar))

  # Compute the eigen decomposition of the covariance matrix, and
  # return the top projection_dims eigen vectors, represented as columns of
  # the projection matrix.
  eigvals, eigvecs = np.linalg.eig(saned_covar)

  topk_indices = eigvals.argsort()[::-1][:projection_dims] 
  topk_indices = np.reshape(topk_indices, [projection_dims])

  # Gather and return the corresponding eigenvectors.
  return np.transpose(np.take(np.transpose(eigvecs), topk_indices, axis=0)), rdp_budget