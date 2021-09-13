from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import json
import sys
from keras.datasets import cifar10
from ops import *
from utils import *
from rdp_utils import *
from pate_core import *
import pickle
from keras.utils import np_utils
# import pandas as pd
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import scipy
from dp_pca import ComputeDPPrincipalProjection
from sklearn.random_projection import GaussianRandomProjection
from utils import pp, visualize, to_json, show_all_variables, mkdir
from gen_data import batch2str
from PIL import Image


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
    assert (int(teacher_id) < int(nb_teachers))

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    # Compute start, end indices of partition
    start = teacher_id * batch_len
    end = (teacher_id + 1) * batch_len

    # Slice partition off
    partition_data = data[start:end]
    if labels is not None:
        partition_labels = labels[start:end]
    else:
        partition_labels = None

    return partition_data, partition_labels


def evenly_partition_dataset(data, labels, nb_teachers):
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

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    nclasses = len(labels[0])
    print("Start Index Selection")
    data_sel = [data[labels[:, j] == 1] for j in range(nclasses)]
    print("End Index Selection")
    i = 0
    data_sel_id = [0] * len(labels[0])
    partition_data = []
    partition_labels = []

    while True:
        partition_data.append(data_sel[i][data_sel_id[i]])
        partition_labels.append(np_utils.to_categorical(i, nclasses))

        if len(partition_data) == batch_len:
            partition_data = np.asarray(partition_data)
            partition_labels = np.asarray(partition_labels)
            yield partition_data, partition_labels
            partition_data = []
            partition_labels = []

        data_sel_id[i] += 1
        if data_sel_id[i] == len(data_sel[i]):
            data_sel_id[i] = 0
        i = (i + 1) % nclasses



def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


class DCGAN(object):
    def __init__(self, sess, input_height=32, input_width=32, crop=False,
                 batch_size=64, sample_num=64, output_height=32, output_width=32,
                 y_dim=10, z_dim=100, gf_dim=64, df_dim=32, sample_step=800,
                 gfc_dim=1024, dfc_dim=256, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, teacher_dir=None, generator_dir=None,
                 sample_dir=None, data_dir='./data', batch_teachers=10, teachers_batch=2,
                 orders=None,
                 thresh=None, dp_delta=1e-5, pca=False, pca_dim=5, non_private=False, random_proj=False, wgan=False,
                 wgan_scale=10, small=False, config=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
          batch_teachers:  Number of teacher models in one batch. Default 10.
          teachers_batch:  Batches of training teacher models. Default 1.
        """
        self.config = config
        self.small = small
        self.wgan = wgan
        self.wgan_scale = wgan_scale

        self.sample_step = sample_step
        self.pca = pca
        self.pca_dim = pca_dim
        self.random_proj = random_proj

        self.dp_eps_list = []
        self.rdp_eps_list = []
        self.rdp_order_list = []
        self.dp_eps_list_dept = []
        self.rdp_eps_list_dept = []
        self.rdp_order_list_dept = []
        self.thresh = thresh
        self.dp_delta = dp_delta
        self.sample_dir = sample_dir
        self.dataset = dataset_name
        self.batch_teachers = batch_teachers
        self.teachers_batch = teachers_batch
        self.overall_teachers = batch_teachers * teachers_batch

        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.y_dim = y_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.teacher_dir = teacher_dir
        self.generator_dir = generator_dir
        self.data_dir = data_dir

        if orders is not None:
            self.orders = np.asarray(orders)
        else:
            self.orders = np.hstack([1.1, np.arange(2, config.orders)])

        self.rdp_counter = np.zeros(self.orders.shape)
        self.rdp_counter_dept = np.zeros(self.orders.shape)

        # Load the dataset, ignore test data for now
        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
            self.grayscale = (self.c_dim == 1)
            self.input_height = self.input_width = 28
            self.output_height = self.output_width = 28

        elif self.dataset_name == 'fashion_mnist':
            self.data_X, self.data_y = self.load_fashion_mnist()
            self.c_dim = self.data_X[0].shape[-1]
            # = (self.c_dim == 1)
            self.input_height = self.input_width = 28
            self.output_height = self.output_width = 28
            if self.config.random_label:
                np.random.shuffle(self.data_y)

        elif self.dataset_name == 'cifar':
            self.data_X, self.data_y = self.load_cifar()
            self.c_dim = self.data_X[0].shape[-1]
            self.grayscale = (self.c_dim == 3)

        elif self.dataset_name == 'cifar-ae':
            _, self.data_y = self.load_cifar()
            import joblib
            self.data_X = joblib.load('cifar10_ae_trn_2d.pkl')
            self.input_height = self.output_height = 24
            self.input_width = self.output_width = 16
            self.c_dim = 1
            self.grayscale = (self.c_dim == 3)

        elif self.dataset_name == 'cifar-ae-hd':
            _, self.data_y = self.load_cifar()
            import joblib
            self.data_X = joblib.load('cifar10_ae_trn_hd_2d.pkl')
            self.input_height = self.output_height = 16
            self.input_width = self.output_width = 16
            self.c_dim = 3
            self.grayscale = (self.c_dim == 3)

        elif self.dataset_name == 'cifar-ae-hd-hu':
            _, self.data_y = self.load_cifar()
            import joblib
            self.data_X = joblib.load('cifar10_ae_trn_hd_hu_2d.pkl')
            self.input_height = self.output_height = 16
            self.input_width = self.output_width = 16
            self.c_dim = 3
            self.grayscale = (self.c_dim == 3)

        elif self.dataset_name == 'cinic-ae-hd-hu':
            import joblib
            self.data_X = joblib.load('cinic10_ae_trn_hd_hu_2d.pkl')
            self.data_y = joblib.load('cinic10_y_trn.pkl')
            self.input_height = self.output_height = 16
            self.input_width = self.output_width = 16
            self.c_dim = 3
            self.grayscale = (self.c_dim == 3)

        elif self.dataset_name == 'cinic-ae-20d':
            import joblib
            self.data_X = joblib.load('cinic10_ae_trn_20d.pkl')
            self.data_y = joblib.load('cinic10_y_trn_20d.pkl')
            self.input_height = self.output_height = 2
            self.input_width = self.output_width = 2
            self.c_dim = 5
            self.grayscale = (self.c_dim == 5)

        elif 'small-celebA-gender' in self.dataset_name:
            mode = self.dataset_name.split('-')[-1]
            self.y_dim = 2
            self.input_size = self.input_height = self.input_width = 32
            self.output_size = self.output_height = self.output_width = 32
            self.data_X, self.data_y = self.load_small_celebA_gender(mode)
            self.c_dim = self.data_X[0].shape[-1]
            self.grayscale = (self.c_dim == 1)

            if self.config.random_label:
                np.random.shuffle(self.data_y)

        elif 'celebA-hair' in self.dataset_name:
            mode = self.dataset_name.split('-')[-1]
            self.y_dim = 3
            self.input_size = self.input_height = self.input_width = 64
            self.output_size = self.output_height = self.output_width = 64
            self.data_X, self.data_y = self.load_batch_celebA_hair(mode)
            self.c_dim = self.data_X[0].shape[-1]
            self.grayscale = (self.c_dim == 1)

            if self.config.random_label:
                np.random.shuffle(self.data_y)

        elif 'celebA-gender' in self.dataset_name:
            mode = self.dataset_name.split('-')[-1]
            self.y_dim = 2
            self.input_size = self.input_height = self.input_width = 64
            self.output_size = self.output_height = self.output_width = 64
            self.data_X, self.data_y = self.load_batch_celebA_gender(mode)
            self.c_dim = self.data_X[0].shape[-1]
            self.grayscale = (self.c_dim == 1)

            if self.config.random_label:
                np.random.shuffle(self.data_y)

        elif self.dataset_name == 'cinic':
            self.data_X, self.data_y = self.load_cinic()
            self.c_dim = self.data_X[0].shape[-1]
            self.grayscale = (self.c_dim == 3)

        elif self.dataset_name == 'stl':
            self.data_X, self.data_y = self.load_stl()
            self.c_dim = self.data_X[0].shape[-1]
            self.grayscale = (self.c_dim == 3)


        elif 'isolet' in self.dataset_name:
            self.data_X, self.data_y = self.load_isolet()
            self.train_size, self.input_size = self.data_X.shape
            self.output_size = self.input_size
            # self.y_dim = None
            # self.crop = False
            if self.pca_dim > self.input_size:
                self.pca_dim = self.input_size

        elif 'fire-small' in self.dataset_name:
            self.data_X = self.load_fire_data()
            self.data_y = None
            self.train_size, self.input_size = self.data_X.shape
            self.output_size = self.input_size
            self.y_dim = None
            self.crop = False
            if self.pca_dim > self.input_size:
                self.pca_dim = self.input_size
        elif 'census' in self.dataset_name:
            self.data_X = self.load_census_data()
            self.data_y = None
            self.train_size, self.input_size = self.data_X.shape
            self.output_size = self.input_size
            self.y_dim = None
            self.crop = False
            if self.pca_dim > self.input_size:
                self.pca_dim = self.input_size
        else:
            raise Exception("Check value of dataset flag")

        print("Dataset load finished!")
        self.train_data_list = []
        self.train_label_list = []

        # if non_private:
        #     for i in range(self.overall_teachers):
        #         partition_data, partition_labels = partition_dataset(self.data_X, self.data_y, 1, i)
        #         self.train_data_list.append(partition_data)
        #         self.train_label_list.append(partition_labels)
        # else:
        from collections import defaultdict
        from tqdm import tqdm
        self.save_dict = defaultdict(lambda: False)
        # stats = []
        if config.shuffle:
            gen = evenly_partition_dataset(self.data_X, self.data_y, self.overall_teachers)
            for i in tqdm(range(self.overall_teachers)):
                partition_data, partition_labels = next(gen)
                self.train_data_list.append(partition_data)
                self.train_label_list.append(partition_labels)
                # stats.append(np.average(partition_labels, axis=0))
                # print(stats[-1])
        else:
            for i in tqdm(range(self.overall_teachers)):
                partition_data, partition_labels = partition_dataset(self.data_X, self.data_y, self.overall_teachers, i)
                self.train_data_list.append(partition_data)
                self.train_label_list.append(partition_labels)
                # stats.append(np.average(partition_labels, axis=0))
                # print(stats[-1])
        # stats = np.asarray(stats)
        # print("avg:", np.average(stats, axis=0))
        # print("max:", np.max(stats, axis=0))
        # print("min:", np.min(stats, axis=0))
        # print(self.train_label_list)
        self.train_size = len(self.train_data_list[0])

        if self.train_size < self.batch_size:
            self.batch_size = self.train_size
            print('adjusted batch size:', self.batch_size)
            # raise Exception("[!] Entire dataset size (%d) is less than the configured batch_size (%d) " % (
            # self.train_size, self.batch_size))
        self.build_model()

    def init_ae(self, load=None):
        if self.config.ae == 'segnet':
            from ae.segnet import segnet
            import torchvision.models as models

            self.ae = segnet(hid_dim=self.config.hid_dim)
            if self.config.hid_dim == 512:
                vgg16 = models.vgg16(pretrained=True)
                self.ae.init_vgg16_params(vgg16)
        elif self.config.ae == 'convnet':
            from ae.convnet import convnet
            self.ae = convnet()
        elif self.config.ae == 'convnetv2':
            from ae.convnetv2 import convnetv2
            self.ae = convnetv2()
        elif self.config.ae == 'convnetv3':
            from ae.convnetv3 import convnetv3
            self.ae = convnetv3()
        elif self.config.ae == 'convnetv4':
            from ae.convnetv4 import convnetv4
            self.ae = convnetv4()
        else:
            raise Exception("Operation not supported")
        if load:
            self.ae.load_state_dict(torch.load(load))

    def init_multitask_ae(self, load=None):
        if self.config.ae == 'segnet':
            from ae.segnet import segnet
            import torchvision.models as models

            self.ae = segnet(hid_dim=self.config.hid_dim)
            if self.config.hid_dim == 512:
                vgg16 = models.vgg16(pretrained=True)
                self.ae.init_vgg16_params(vgg16)
        elif self.config.ae == 'convnet':
            from ae.convnet import convnet, convnet_multitask
            self.ae = convnet_multitask(convnet())
        elif self.config.ae == 'convnetv2':
            from ae.convnetv2 import convnetv2, convnetv2_multitask
            self.ae = convnetv2_multitask(convnetv2())
        elif self.config.ae == 'convnetv3':
            from ae.convnetv3 import convnetv3, convnetv3_multitask
            self.ae = convnetv3_multitask(convnetv3())
        elif self.config.ae == 'convnetv4':
            from ae.convnetv4 import convnetv4, convnetv4_multitask
            self.ae = convnetv4_multitask(convnetv4())
        else:
            raise Exception("Operation not supported")
        if load:
            self.ae.model.load_state_dict(torch.load(load))

    def finetune_ae(self):
        from tensorboardX import SummaryWriter
        from tqdm import tqdm

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_name)
        self.init_multitask_ae(ckpt)
        model = self.ae = self.ae.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.98**x, 1e-6))
        if self.config.loss == 'l2':
            alpha = 20
            criterion = torch.nn.MSELoss()
        elif self.config.loss == 'BCE':
            criterion = torch.nn.BCELoss()
            alpha = 5
        elif self.config.loss == 'l1':
            criterion = torch.nn.L1Loss()
            # alpha = 0
        reconstruct_criterion = torch.nn.MSELoss()
        task_criterion = torch.nn.CrossEntropyLoss()

        ckpt_dir = 'finetune-{}-{}/'.format(self.dataset_name, alpha) + self.config.checkpoint_dir
        x = self.data_X
        y = self.data_y
        writer = SummaryWriter(ckpt_dir)

        if self.dataset_name == 'cifar':
            test_x, test_y = self.load_cifar_test()
        else:
            test_x, test_y = self.load_cinic_test()
        # _, y = torch.max(y, dim=0)
        # _, test_y = torch.max(test_y, dim=0)
        x = np.transpose(x, (0, 3, 2, 1))
        test_x = np.transpose(test_x, (0, 3, 2, 1))
        best_test_loss = np.inf

        for epoch in range(self.config.epoch):
            total_loss = []
            flag = False
            ## training on the stl
            model.train()
            from sklearn.utils import shuffle
            x, y = shuffle(x, y)
            for i in tqdm(range(0, len(x), self.batch_size)):
                batch_x = torch.FloatTensor(x[i:min(i + self.batch_size, len(x))]).to(device)
                batch_y = torch.LongTensor(y[i:min(i + self.batch_size, len(y))]).to(device)
                _, batch_y = torch.max(batch_y, dim=1)
                logits, decoded = model(batch_x)

                task_loss = task_criterion(logits, batch_y)
                reconstruct_loss = criterion(decoded, batch_x)
                loss = task_loss + reconstruct_loss * alpha
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                total_loss += loss.item(),

                writer.add_scalar('ae/loss', loss, i + epoch * len(x))

            total_loss = np.mean(total_loss)
            print("|epoch {} | total loss: {}".format(epoch, total_loss))

            test_loss = []
            test_acc = []
            test_reconstruct = []
            ## testing on
            model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, len(test_x), self.batch_size)):
                    batch_x = torch.tensor(test_x[i:min(i + self.batch_size, len(x))]).to(device)
                    batch_y = torch.LongTensor(test_y[i:min(i + self.batch_size, len(y))]).to(device)
                    _, batch_y = torch.max(batch_y, dim=1)

                    logits, decoded = model(batch_x)
                    _, pred = torch.max(logits, dim=1)
                    accuracy = (batch_y == pred).float().mean().item()
                    reconstruct_loss = reconstruct_criterion(decoded, batch_x)
                    loss = reconstruct_loss * alpha + task_criterion(logits, batch_y)
                    test_loss += loss.item(),
                    test_reconstruct = reconstruct_loss.item(),
                    test_acc += accuracy,

            test_loss = np.mean(test_loss)
            test_acc = np.mean(test_acc)
            test_reconstruct = np.mean(test_reconstruct)
            print("|epoch {} | total test loss: {} | total test accuracy: {} | reconstruct loss: {}".
                  format(epoch, test_loss, test_acc, test_reconstruct))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                ## save model
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model-epoch_{}.pth'.format(epoch)))
            elif epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model-epoch_{}.pth'.format(epoch)))

            scheduler.step()
        writer.close()


    def train_ae(self):
        from tensorboardX import SummaryWriter
        from tqdm import tqdm

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_ae()
        model = self.ae = self.ae.to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
        #                              weight_decay=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.98**x, 1e-6))
        if self.config.loss == 'l2':
            criterion = torch.nn.MSELoss()
        elif self.config.loss == 'BCE':
            criterion = torch.nn.BCELoss()
        elif self.config.loss == 'l1':
            criterion = torch.nn.L1Loss()

        ckpt_dir = 'runs/' + self.config.checkpoint_name
        x = self.data_X
        writer = SummaryWriter(ckpt_dir)

        test_x, test_y = self.load_cifar()
        x = np.transpose(x, (0, 3, 2, 1))
        test_x = np.transpose(test_x, (0, 3, 2, 1))
        best_test_loss = np.inf

        for epoch in range(self.config.epoch):
            total_loss = []
            flag = False
            ## training on the stl
            model.train()
            from sklearn.utils import shuffle
            x = shuffle(x)
            for i in tqdm(range(0, len(x), self.batch_size)):
                batch_x = torch.FloatTensor(x[i:min(i + self.batch_size, len(x))]).to(device)
                outputs = model(batch_x)

                if not flag:
                    flag = True
                    import joblib
                    joblib.dump((outputs, batch_x), os.path.join(ckpt_dir, 'debug.pkl'))
                loss = criterion(outputs, batch_x)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                total_loss += loss.item(),

                writer.add_scalar('ae/loss', loss, i + epoch * len(x))

            total_loss = np.mean(total_loss)
            print("|epoch {} | total loss: {}".format(epoch, total_loss))

            test_loss = []
            ## testing on
            model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, len(test_x), self.batch_size)):
                    batch_x = torch.tensor(test_x[i:min(i + self.batch_size, len(x))]).to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    test_loss += loss.item(),

            test_loss = np.mean(test_loss)
            print("|epoch {} | total test loss: {}".format(epoch, test_loss))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                ## save model
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model-epoch_{}.pth'.format(epoch)))
            elif epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model-epoch_{}.pth'.format(epoch)))

            scheduler.step()
        writer.close()

    ## nonprivate aggregation
    def aggregate_topk(self, output_list, topk, alpha=1e-3):
        flatten_grad = np.asarray([arr.flatten() for arr in output_list])
        flatten_grad_pos = np.array(flatten_grad).clip(0)
        flatten_grad_neg = np.array(flatten_grad).clip(max=0)
        concat_grad = np.hstack((flatten_grad_pos, flatten_grad_neg))
        aggregate_grad = np.abs(np.sum(concat_grad, axis=0))

        if self.config.save_vote and self.epoch_change:
            self.epoch_change = False
            def save_sign_weighted_votes():
                save_dir = os.path.join(self.checkpoint_dir, 'sign_weighted_votes.pkl')
                import joblib
                if os.path.exists(save_dir):
                    votes = joblib.load(save_dir)
                    votes = np.vstack((votes, aggregate_grad))
                    joblib.dump(votes, save_dir)
                    print(aggregate_grad.shape)
                else:
                    joblib.dump(aggregate_grad, save_dir)

            def save_sign_equal_votes():
                save_dir = os.path.join(self.checkpoint_dir, 'sign_equal_votes.pkl')
                voted_arr = np.sum(convert2topk(np.abs(concat_grad), topk), axis=0)
                import joblib
                if os.path.exists(save_dir):
                    votes = joblib.load(save_dir)
                    votes = np.vstack((votes, voted_arr))
                    print(voted_arr.shape)
                    joblib.dump(votes, save_dir)
                else:
                    joblib.dump(voted_arr, save_dir)

            def save_unsign_equal_votes():
                save_dir = os.path.join(self.checkpoint_dir, 'unsign_equal_votes.pkl')
                voted_arr = np.sum(convert2topk(np.abs(flatten_grad), topk), axis=0)
                import joblib
                if os.path.exists(save_dir):
                    votes = joblib.load(save_dir)
                    votes = np.vstack((votes, voted_arr))
                    print(voted_arr.shape)

                    joblib.dump(votes, save_dir)
                else:
                    joblib.dump(voted_arr, save_dir)

            def save_unsign_weighted_votes():
                save_dir = os.path.join(self.checkpoint_dir, 'unsign_weighted_votes.pkl')
                voted_arr = np.sum(np.abs(flatten_grad), axis=0)
                import joblib
                if os.path.exists(save_dir):
                    votes = joblib.load(save_dir)
                    votes = np.vstack((votes, voted_arr))
                    print(voted_arr.shape)
                    joblib.dump(votes, save_dir)
                else:
                    joblib.dump(voted_arr, save_dir)

            save_sign_equal_votes()
            save_sign_weighted_votes()
            save_unsign_equal_votes()
            save_unsign_weighted_votes()

        topk_ind = np.argpartition(aggregate_grad, -topk)[-topk:]
        pos_ind = topk_ind[topk_ind < flatten_grad[0].shape]
        neg_ind = topk_ind[topk_ind >= flatten_grad[0].shape]
        neg_ind -= flatten_grad[0].shape
        sign_grad = np.zeros_like(flatten_grad[0])
        sign_grad[pos_ind] = 1
        sign_grad[neg_ind] = -1

        return alpha * sign_grad.reshape(output_list[0].shape)


    def aggregate_results(self, output_list, config, thresh=None, epoch=None):
        if self.pca:
            res, rdp_budget = gradient_voting_rdp(
                output_list,
                config.step_size,
                config.sigma,
                config.sigma_thresh,
                self.orders,
                pca_mat=self.pca_components,
                thresh=thresh
            )
        elif self.config.mean_kernel:
            from skimage.measure import block_reduce
            from skimage.transform import resize
            arr = np.asarray(output_list)
            mean_arr = block_reduce(arr, block_size=(1, 8, 8, 1), func=np.mean)
            res, rdp_budget = gradient_voting_rdp(
                mean_arr,
                config.step_size,
                config.sigma,
                config.sigma_thresh,
                self.orders,
                thresh=thresh
            )
            res = resize(res, (16, 16, 3), clip=False)
        elif self.config.signsgd:
            # res = self.aggregate_topk(output_list, topk=self.config.topk, alpha=self.config.learning_rate)
            # rdp_budget = 0
            if self.config.save_vote and self.epoch_change:
                self.epoch_change = False
                ablation_test_on_different_k(output_list, self.checkpoint_dir, self.epoch)
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = signsgd_aggregate(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
        elif self.config.signsgd_nothresh:
            # res = self.aggregate_topk(output_list, topk=self.config.topk, alpha=self.config.learning_rate)
            # rdp_budget = 0
            if self.config.save_vote and self.epoch_change:
                self.epoch_change = False
                ablation_test_on_different_k(output_list, self.checkpoint_dir, self.epoch)
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = signsgd_aggregate_no_thresh(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
        elif self.config.sketchsgd:
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = sketchtopk_aggregate(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
        elif self.config.klevelsgd:
            # res = self.aggregate_topk(output_list, topk=self.config.topk, alpha=self.config.learning_rate)
            # rdp_budget = 0
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = k_level_sgd_aggregate(output_list, config.sigma, self.orders, config.klevel, config.thresh,
                                                alpha=config.learning_rate, b=b)
        elif self.config.signsgd_dept:
            if self.config.save_vote and self.epoch_change:
                self.epoch_change = False
                ablation_test_on_different_k(output_list, self.checkpoint_dir, self.epoch)
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget, dept_rdp_budget = signsgd_aggregate_dept(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
            return res, rdp_budget, dept_rdp_budget
        elif self.random_proj:
            orig_dim = 1
            for dd in self.image_dims:
                orig_dim = orig_dim * dd

            if epoch is not None:
                proj_dim = min(epoch + 1, self.pca_dim)
            else:
                proj_dim = self.pca_dim

            n_data = output_list[0].shape[0]
            if config.proj_mat > 1:
                proj_dim_ = proj_dim // config.proj_mat
                n_data_ = n_data // config.proj_mat
                orig_dim_ = orig_dim // config.proj_mat
                print("n_data:", n_data)
                print("orig_dim:", orig_dim)
                transformers = [GaussianRandomProjection(n_components=proj_dim_) for _ in range(config.proj_mat)]
                for transformer in transformers:
                    transformer.fit(np.zeros([n_data_, orig_dim_]))
                    print(transformer.components_.shape)
                proj_matrices = [np.transpose(transformer.components_) for transformer in transformers]
                res, rdp_budget = gradient_voting_rdp_multiproj(
                    output_list,
                    config.step_size,
                    config.sigma,
                    config.sigma_thresh,
                    self.orders,
                    pca_mats=proj_matrices,
                    thresh=thresh
                )
            else:
                transformer = GaussianRandomProjection(n_components=proj_dim)
                transformer.fit(np.zeros([n_data, orig_dim]))  # only the shape of output_list[0] is used
                proj_matrix = np.transpose(transformer.components_)

                # proj_matrix = np.random.normal(loc=np.zeros([orig_dim, proj_dim]), scale=1/float(proj_dim), size=[orig_dim, proj_dim])
                res, rdp_budget = gradient_voting_rdp(
                    output_list,
                    config.step_size,
                    config.sigma,
                    config.sigma_thresh,
                    self.orders,
                    pca_mat=proj_matrix,
                    thresh=thresh
                )
        else:
            res, rdp_budget = gradient_voting_rdp(output_list, config.step_size, config.sigma, config.sigma_thresh,
                                                  self.orders, thresh=thresh)
        return res, rdp_budget

    def non_private_aggregation(self, output_list, config):
        # TODO update nonprivate aggregation
        sum_arr = np.zeros(output_list[0].shape)
        for arr in output_list:
            sum_arr += arr
        return sum_arr / len(output_list)

    def load_fire_data(self):
        dataset_name = os.path.join(self.data_dir, self.dataset_name)
        dataset_name += '.csv'
        X = np.loadtxt(dataset_name)
        seed = 307
        np.random.seed(seed)
        np.random.shuffle(X)
        return X

    def load_census_data(self):
        dataset_name = os.path.join(self.data_dir, self.dataset_name)
        dataset_name += '.pkl'
        with open(dataset_name, "rb") as f:
            X = pickle.load(f)
        seed = 37
        np.random.seed(seed)
        np.random.shuffle(X)
        return X

    def load_isolet(self):
        dataset_name = os.path.join(self.data_dir, self.dataset_name)
        dataset_name += '.csv'
        X = np.loadtxt(dataset_name)
        # print(X.shape)
        seed = 37
        np.random.seed(seed)
        np.random.shuffle(X)
        X = np.hsplit(X, [-1])
        x = X[0]
        # print(X.shape)
        y = X[1]
        # print(y.shape)
        y = np_utils.to_categorical(y, 2)
        # print(y.shape)
        return x, y

    def load_cifar(self):
        # dataset_name = os.path.join(self.data_dir, self.dataset_name)
        # dataset_name += '.csv'
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np_utils.to_categorical(y_train, 10)
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_train = x_train.astype('float32') / 255.
        return x_train, y_train

    def load_cifar_test(self):
        # dataset_name = os.path.join(self.data_dir, self.dataset_name)
        # dataset_name += '.csv'
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_test = np_utils.to_categorical(y_test, 10)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
        x_test = x_test.astype('float32') / 255.
        return x_test, y_test

    def load_stl(self):

        path_to_data = '../data/stl10_binary/unlabeled_X.bin'
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)

            # We force the data into 3x96x96 chunks, since the
            # images are stored in "column-major order", meaning
            # that "the first 96*96 values are the red channel,
            # the next 96*96 are green, and the last are blue."
            # The -1 is since the size of the pictures depends
            # on the input file, and this way numpy determines
            # the size on its own.

            images = np.reshape(everything, (-1, 3, 96, 96))

            # Now transpose the images into a standard image format
            # readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the shuffle
            # if you will use a learning algorithm like CNN, since they like
            # their channels separated.
            images = np.transpose(images, (0, 3, 2, 1))
            X_resized = np.zeros((100000, 32, 32, 3))
            for i in range(0, 100000):
                img = images[i]
                img = Image.fromarray(img)
                img = np.array(img.resize((32, 32), Image.BICUBIC))  # 修改分辨率，再转为array类
                X_resized[i, :, :, :] = img

            y = np.random.randint(10, size=(100000, 1))
            y = np_utils.to_categorical(y, 10)
            X_resized /= 255
            print(X_resized)
            return X_resized, y

    def load_cinic(self):
        import torchvision
        import torchvision.transforms as transforms
        from keras.utils import np_utils

        cinic_directory = '../data/cinic'
        # cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        # cinic_std = [0.24205776, 0.23828046, 0.25874835]
        image_folder = torchvision.datasets.ImageFolder(cinic_directory + '/train/',
                                                        # transform=transforms.Compose([transforms.ToTensor(),
                                                        # transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
                                                        transform=transforms.ToTensor())
        cinic_train = torch.utils.data.DataLoader(image_folder, batch_size=180000, shuffle=True)
        print(image_folder.class_to_idx)
        for batch_ndx, sample in enumerate(cinic_train):
            x = np.asarray(sample[0])
            y = np.asarray(sample[1])
            x = np.transpose(x, [0, 2, 3, 1])
            y = np_utils.to_categorical(y, 10)
            return x, y

    def load_cinic_test(self):
        import torchvision
        import torchvision.transforms as transforms
        from keras.utils import np_utils

        cinic_directory = '../data/cinic'
        # cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        # cinic_std = [0.24205776, 0.23828046, 0.25874835]
        image_folder = torchvision.datasets.ImageFolder(cinic_directory + '/test/',
                                                        # transform=transforms.Compose([transforms.ToTensor(),
                                                        # transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
                                                        transform=transforms.ToTensor())
        cinic_train = torch.utils.data.DataLoader(image_folder, batch_size=90000, shuffle=True)
        print(image_folder.class_to_idx)
        for batch_ndx, sample in enumerate(cinic_train):
            x = np.asarray(sample[0])
            y = np.asarray(sample[1])
            x = np.transpose(x, [0, 2, 3, 1])
            y = np_utils.to_categorical(y, 10)
            return x, y

    def load_celebA_gender(self, mode='train'):
        celebA_directory = '../../data/celebA/'
        import joblib

        if mode == 'train':
            train_x = joblib.load(celebA_directory + 'celebA-trn-x-lg-ups.pkl')
            train_y = joblib.load(celebA_directory + 'celebA-trn-gender-lg-ups.pkl')
            train_y = np_utils.to_categorical(train_y, 2)
            val_x = joblib.load(celebA_directory + 'celebA-val-x-lg-ups.pkl')
            val_y = joblib.load(celebA_directory + 'celebA-val-gender-lg-ups.pkl')
            val_y = np_utils.to_categorical(val_y, 2)
            return np.vstack((train_x, val_x)), np.vstack((train_y, val_y))
        elif mode == 'val':
            val_x = joblib.load(celebA_directory + 'celebA-val-x-lg-ups.pkl')
            val_y = joblib.load(celebA_directory + 'celebA-val-gender-lg-ups.pkl')
            val_y = np_utils.to_categorical(val_y, 2)
            return val_x, val_y
        elif mode == 'tst':
            tst_x = joblib.load(celebA_directory + 'celebA-tst-x.pkl')
            tst_y = joblib.load(celebA_directory + 'celebA-tst-gender.pkl')
            tst_y = np_utils.to_categorical(tst_y, 2)
            return tst_x, tst_y
        else:
            raise Exception("Mode {} Not support".format(mode))

    def load_batch_celebA_gender(self, mode='train'):
        celebA_directory = '../../data/celebA/'
        import joblib
        from tqdm import tqdm

        if mode == 'train':
            train_x = np.zeros((189018 + 22818, 64, 64, 3))
            dim = 0
            for i in tqdm(range(20)):
                x = joblib.load(celebA_directory + f'celebA-trn-x-lg-ups-{i}.pkl')
                train_x[dim: dim + len(x)] = x
                dim += len(x)
            train_y = joblib.load(celebA_directory + 'celebA-trn-gender-lg-ups.pkl')
            train_y = np_utils.to_categorical(train_y, 2)
            for i in tqdm(range(10)):
                x = joblib.load(celebA_directory + f'celebA-val-x-lg-ups-{i}.pkl')
                train_x[dim: dim + len(x)] = x
                dim += len(x)
            val_y = joblib.load(celebA_directory + 'celebA-val-gender-lg-ups.pkl')
            val_y = np_utils.to_categorical(val_y, 2)
            print("Load train x finished!")
            return train_x, np.vstack((train_y, val_y))
        elif mode == 'val':
            val_x = []
            for i in tqdm(range(10)):
                val_x.append(joblib.load(celebA_directory + f'celebA-val-x-lg-ups-{i}.pkl'))
            val_y = joblib.load(celebA_directory + 'celebA-val-gender-lg-ups.pkl')
            val_y = np_utils.to_categorical(val_y, 2)
            return np.vstack(val_x), val_y
        elif mode == 'tst':
            tst_x = joblib.load(celebA_directory + 'celebA-tst-x.pkl')
            tst_y = joblib.load(celebA_directory + 'celebA-tst-gender.pkl')
            tst_y = np_utils.to_categorical(tst_y, 2)
            return tst_x, tst_y
        else:
            raise Exception("Mode {} Not support".format(mode))

    def load_batch_celebA_hair(self, mode='trn'):
        celebA_directory = '../../data/celebA/'
        import joblib
        from tqdm import tqdm

        if mode == 'trn':
            train_x = np.zeros((127713, 64, 64, 3))
            dim = 0
            for i in tqdm(range(10)):
                x = joblib.load(celebA_directory + f'celeb-trn-ups-hair-x-{i}.pkl')
                train_x[dim: dim + len(x)] = x
                dim += len(x)
            train_y = joblib.load(celebA_directory + 'celeb-trn-ups-hair-y.pkl')
            train_y = np_utils.to_categorical(train_y, 3)
            for i in tqdm(range(10)):
                x = joblib.load(celebA_directory + f'celeb-val-ups-hair-x-{i}.pkl')
                train_x[dim: dim + len(x)] = x
                dim += len(train_x[-1])
            val_y = joblib.load(celebA_directory + 'celeb-val-ups-hair-y.pkl')
            val_y = np_utils.to_categorical(val_y, 3)
            print("Load train x finished!")
            return train_x, np.vstack((train_y, val_y))
        elif mode == 'val':
            for i in tqdm(range(10)):
                val_x = []
                val_x.append(joblib.load(celebA_directory + f'celeb-val-ups-hair-x-{i}.pkl'))
            val_y = joblib.load(celebA_directory + 'celeb-val-ups-hair-y.pkl')
            val_y = np_utils.to_categorical(val_y, 3)
            return np.vstack(val_x), val_y
        elif mode == 'tst':
            tst_x = joblib.load(celebA_directory + 'celeb-tst-ups-hair-x.pkl')
            tst_y = joblib.load(celebA_directory + 'celeb-tst-ups-hair-y.pkl')
            tst_y = np_utils.to_categorical(tst_y, 3)
            return tst_x, tst_y
        else:
            raise Exception("Mode {} Not support".format(mode))

    def load_celebA_hair(self, mode='trn'):
        celebA_directory = '../../data/celebA/'
        import joblib

        if mode == 'trn':
            train_x = joblib.load(celebA_directory + 'celeb-trn-ups-hair-x.pkl')
            train_y = joblib.load(celebA_directory + 'celeb-trn-ups-hair-y.pkl')
            train_y = np_utils.to_categorical(train_y, 3)
            val_x = joblib.load(celebA_directory + 'celeb-val-ups-hair-x.pkl')
            val_y = joblib.load(celebA_directory + 'celeb-val-ups-hair-y.pkl')
            val_y = np_utils.to_categorical(val_y, 3)
            return np.vstack((train_x, val_x)), np.vstack((train_y, val_y))
        elif mode == 'val':
            val_x = joblib.load(celebA_directory + 'celeb-val-ups-hair-x.pkl')
            val_y = joblib.load(celebA_directory + 'celeb-val-ups-hair-y.pkl')
            val_y = np_utils.to_categorical(val_y, 3)
            return val_x, val_y
        elif mode == 'tst':
            tst_x = joblib.load(celebA_directory + 'celeb-tst-ups-hair-x.pkl')
            tst_y = joblib.load(celebA_directory + 'celeb-tst-ups-hair-y.pkl')
            tst_y = np_utils.to_categorical(tst_y, 3)
            return tst_x, tst_y
        else:
            raise Exception("Mode {} Not support".format(mode))

    def load_small_celebA_gender(self, mode='train'):
        celebA_directory = '../../data/celebA/'
        import joblib

        if mode == 'train':
            train_x = joblib.load(celebA_directory + 'celebA-trn-x-small-ups.pkl')
            train_y = joblib.load(celebA_directory + 'celebA-trn-gender-ups.pkl')
            train_y = np_utils.to_categorical(train_y, 2)
            val_x = joblib.load(celebA_directory + 'celebA-val-x-small-ups.pkl')
            val_y = joblib.load(celebA_directory + 'celebA-val-gender-ups.pkl')
            val_y = np_utils.to_categorical(val_y, 2)
            return np.vstack((train_x, val_x)), np.vstack((train_y, val_y))
        elif mode == 'val':
            val_x = joblib.load(celebA_directory + 'celebA-val-x-small-ups.pkl')
            val_y = joblib.load(celebA_directory + 'celebA-val-gender-ups.pkl')
            val_y = np_utils.to_categorical(val_y, 2)
            return val_x, val_y
        elif mode == 'tst':
            tst_x = joblib.load(celebA_directory + 'celebA-tst-x-small.pkl')
            tst_y = joblib.load(celebA_directory + 'celebA-tst-gender.pkl')
            tst_y = np_utils.to_categorical(tst_y, 2)
            return tst_x, tst_y
        else:
            raise Exception("Mode {} Not support".format(mode))

    def load_fashion_mnist(self):
        data_dir = os.path.join(self.data_dir, self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.int)

        # fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        # loaded = np.fromfile(file=fd,dtype=np.uint8)
        # teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        # fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        # loaded = np.fromfile(file=fd,dtype=np.uint8)
        # teY = loaded[8:].reshape((10000)).astype(np.int)

        trY = np.asarray(trY)
        # teY = np.asarray(teY)

        # X = np.concatenate((trX, teX), axis=0)
        # y = np.concatenate((trY, teY), axis=0).astype(np.int)
        X = trX
        y = trY.astype(np.int)

        seed = 307
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    def load_mnist(self):
        data_dir = os.path.join(self.data_dir, self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.int)

        trY = np.asarray(trY)

        X = trX
        y = trY.astype(np.int)

        seed = 307
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0
        return X / 255., y_vec

    def build_model(self):
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + [self.input_height, self.input_width, self.c_dim], name='real_images')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        self.image_dims = image_dims

        inputs = self.inputs
        if self.crop:
            inputs = tf.image.resize_image_with_crop_or_pad(inputs, target_height=self.output_height,
                                                            target_width=self.output_width)

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        if 'slt' in self.dataset_name or 'cifar' in self.dataset_name:
            self.G_sum = image_summary("G", self.G, max_outputs=10)

        self.updated_img = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='updated_img')
        self.g_loss = tf.reduce_sum(tf.square(self.updated_img - self.G))

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

        self.teachers_list = []
        for i in range(self.batch_teachers):
            with tf.variable_scope("teacher%d" % i) as scope:
                D, D_logits = self.discriminator(inputs, self.y)

                scope.reuse_variables()
                D_, D_logits_ = self.discriminator(self.G, self.y)

                if self.wgan:
                    # Use WassersteinGAN loss with gradient penalty. Reference: https://github.com/jiamings/wgan/blob/master/wgan_v2.py
                    # Calculate interpolation of real and fake image
                    if 'mnist' in self.dataset_name:
                        alpha = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
                        alpha = tf.tile(alpha, tf.constant([1, self.input_height, self.input_width, self.c_dim]))
                    else:
                        alpha = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
                        alpha = tf.tile(alpha, tf.constant([1, self.input_size]))

                    x_hat = tf.math.multiply(alpha, inputs) + tf.math.multiply((1 - alpha), self.G)
                    _, d_hat = self.discriminator(x_hat, self.y)

                    # Calculate gradient penalty for wgan
                    ddx = tf.gradients(d_hat, x_hat)[0]
                    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
                    ddx = tf.reduce_mean(tf.square(ddx - 1.0) ** 2 * self.wgan_scale)

            if self.wgan:
                teacher = {
                    'd_loss': tf.reduce_mean(D_logits_) - tf.reduce_mean(D_logits) + ddx,
                    'g_loss': -tf.reduce_mean(D_logits_),
                }
            else:
                teacher = {
                    'd_loss': tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D))) + \
                              tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_))),
                    'g_loss': tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_))),
                }

            teacher.update({
                'd_loss_sum': scalar_summary("d_loss_%d" % i, teacher['d_loss']),
                'g_loss_sum': scalar_summary("g_loss_%d" % i, teacher['g_loss']),
            })

            # calculate the change in the images that would minimize generator loss
            teacher['img_grads'] = -tf.gradients(teacher['g_loss'], self.G)[0]

            if 'slt' in self.dataset_name:
                teacher['img_grads_sum'] = image_summary("img_grads", teacher['img_grads'], max_outputs=10)

            self.teachers_list.append(teacher)

        t_vars = tf.trainable_variables()
        g_list = tf.global_variables()
        add_save = [g for g in g_list if "moving_mean" in g.name]
        add_save += [g for g in g_list if "moving_variance" in g.name]

        self.save_vars = t_vars + add_save

        self.d_vars = []
        for i in range(self.batch_teachers):
            self.d_vars.append([var for var in t_vars if 'teacher%d' % i in var.name])
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_save_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_save_vars = [var for var in t_vars if 'd_' in var.name]
        # print(self.d_save_vars)
        print(self.save_vars)
        # self.d_save_vars = {'k': v for k, v in zip(self.d_save_vars, self.d_save_vars)}
        self.saver = tf.train.Saver(max_to_keep=5, var_list=self.save_vars)
        self.saver_g = tf.train.Saver(max_to_keep=5, var_list=self.g_save_vars)
        self.saver_d = tf.train.Saver(max_to_keep=self.teachers_batch, var_list=self.d_save_vars)

    def get_random_labels(self, batch_size):
        # print(self.y_dim)
        y_vec = np.zeros((batch_size, self.y_dim), dtype=np.float)
        y = np.random.randint(0, self.y_dim, batch_size)

        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return y_vec

    def train_together(self, config):
        print("Training teacher models and student model together...")

        if not config.non_private:
            assert len(self.train_data_list) == self.overall_teachers
        else:
            print(str(len(self.train_data_list)))

        configs = {
            'sigma': config.sigma,
            'sigma_thresh': config.sigma_thresh,
            'pca': self.pca,
            'pca_sigma': config.pca_sigma,
            'step_size': config.step_size,
            'batch_teachers': self.batch_teachers,
            'g_step': config.g_step,
            'pca_dim': self.pca_dim,
        }

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.teacher_dir):
            os.makedirs(self.teacher_dir)

        with open(os.path.join(self.checkpoint_dir, 'configs.json'), 'w') as fp:
            json.dump(configs, fp)

        if self.pca:
            data = self.data_X.reshape([self.data_X.shape[0], -1])
            self.pca_components, rdp_budget = ComputeDPPrincipalProjection(
                data,
                self.pca_dim,
                self.orders,
                config.pca_sigma,
            )
            self.rdp_counter += rdp_budget

        d_optim_list = []

        for i in range(self.batch_teachers):
            d_optim_list.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(
                self.teachers_list[i]['d_loss'], var_list=self.d_vars[i]))

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss,
                                                                                            var_list=self.g_vars)

        if not config.pretrain:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
        else:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            self.load_pretrain(config.checkpoint_dir)
            # data = self.gen_data(5000)
            # output_dir = os.path.join(self.checkpoint_dir, self.sample_dir)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # filename = 'private.data_epoch_' + str(-1) + '.pkl'
            # outfile = os.path.join(output_dir, filename)
            # mkdir(output_dir)
            # with open(outfile, 'wb') as f:
            #     pickle.dump(data, f)
            # current_scope = tf.contrib.framework.get_name_scope()
            # with tf.variable_scope(current_scope, reuse=True):
            #     biases = tf.get_variable("teacher0/d_h0_conv/biases")
            #     biases = tf.Print(biases, [biases])
            #     self.sess.run(biases)

        if 'slt' in self.dataset_name:
            self.g_sum = merge_summary([self.z_sum, self.G_sum, self.g_loss_sum])
        else:
            self.g_sum = merge_summary([self.z_sum, self.g_loss_sum])

        self.d_sum_list = []

        for i in range(self.batch_teachers):
            teacher = self.teachers_list[i]
            if 'slt' in self.dataset_name:
                self.d_sum_list.append(
                    merge_summary([teacher['d_loss_sum'], teacher['g_loss_sum'], teacher['img_grads_sum']]))
            else:
                self.d_sum_list.append(merge_summary([teacher['d_loss_sum'], teacher['g_loss_sum']]))

        self.writer = SummaryWriter(os.path.join(self.checkpoint_dir, "logs"), self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        counter = 0
        start_time = time.time()

        self.save_d(self.teacher_dir, 0, -1)
        for epoch in xrange(config.epoch):
            self.epoch_change = True
            self.epoch = epoch
            print("----------------epoch: %d --------------------" % epoch)
            print("-------------------train-teachers----------------")
            batch_idxs = int(min(self.train_size, config.train_size) // self.batch_size)
            # The idex of each batch
            print("Train %d idxs" % batch_idxs)
            for idx in xrange(0, batch_idxs):

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                errD = 0
                # train teacher models in batches, teachers_batch: how many batches of teacher
                for batch_num in range(self.teachers_batch):
                    if self.teachers_batch > 1:
                        could_load, checkpoint_counter = self.load_d(self.teacher_dir, epoch=epoch,
                                                                     batch_num=batch_num)
                        if could_load:
                            counter = checkpoint_counter
                            print("load sucess_this_epoch")
                        else:
                            print('fail_1')
                            could_load, checkpoint_counter = self.load_d(self.teacher_dir, epoch=epoch - 1,
                                                                         batch_num=batch_num)
                            if could_load:
                                counter = checkpoint_counter
                                print("load sucess_previous_epoch")
                            else:
                                print('fail_2')
                                could_load, checkpoint_counter = self.load_d(self.teacher_dir, epoch=0,
                                                                             batch_num=-1)

                    # train each teacher in this batch, batch_teachers: how many teacher in a batch
                    for teacher_id in range(self.batch_teachers):
                        # print("Training teacher model %d" % teacher_id)
                        # data_X = self.data_X if config.non_private else self.train_data_list[teacher_id+batch_num*self.batch_teachers]
                        data_X = self.train_data_list[teacher_id + batch_num * self.batch_teachers]

                        batch_idx = range(idx * self.batch_size, (idx + 1) * self.batch_size)
                        batch_images = data_X[batch_idx]

                        for k in range(config.d_step if epoch > 0 or config.pretrain_teacher == 0 else config.pretrain_teacher):
                            if self.y is not None:
                                # data_y = self.data_y if config.non_private else self.train_label_list[teacher_id+batch_num*self.batch_teachers]
                                data_y = self.train_label_list[teacher_id + batch_num * self.batch_teachers]
                                # print(data_y.shape)
                                batch_labels = data_y[batch_idx]

                                _, summary_str = self.sess.run([d_optim_list[teacher_id], self.d_sum_list[teacher_id]],
                                                               feed_dict={
                                                                   self.inputs: batch_images,
                                                                   self.z: batch_z,
                                                                   self.y: batch_labels,
                                                               })

                                self.writer.add_summary(summary_str, epoch)

                                err = self.teachers_list[teacher_id]['d_loss'].eval({
                                    self.z: batch_z,
                                    self.inputs: batch_images,
                                    self.y: batch_labels,
                                })
                                # print(str(batch_num*self.batch_teachers + teacher_id) + "loss:"+str(err))
                                errD += err
                            else:
                                _, summary_str = self.sess.run([d_optim_list[teacher_id], self.d_sum_list[teacher_id]],
                                                               feed_dict={
                                                                   self.inputs: batch_images,
                                                                   self.z: batch_z,
                                                               })

                                self.writer.add_summary(summary_str, epoch)

                                err = self.teachers_list[teacher_id]['d_loss'].eval({
                                    self.z: batch_z,
                                    self.inputs: batch_images,
                                })
                                # print(str(batch_num * self.batch_teachers + teacher_id) + "d_loss:" + str(err))
                                errD += err

                    self.save_d(self.teacher_dir, epoch, batch_num)

                # print("------------------train-generator-------------------")
                for k in range(config.g_step):
                    errG = 0
                    img_grads_list = []
                    if self.y is not None:
                        batch_labels = self.get_random_labels(self.batch_size)
                        for batch_num in range(self.teachers_batch):
                            if self.teachers_batch > 1:
                                could_load, checkpoint_counter = self.load_d(self.teacher_dir, epoch=epoch,
                                                                             batch_num=batch_num)
                                if could_load:
                                    counter = checkpoint_counter
                                    print("load sucess")
                                else:
                                    print('fail')

                            for teacher_id in range(self.batch_teachers):
                                img_grads = self.sess.run(self.teachers_list[teacher_id]['img_grads'],
                                                          feed_dict={
                                                              self.z: batch_z,
                                                              self.y: batch_labels,
                                                          })
                                img_grads_list.append(img_grads)

                        old_img = self.sess.run(self.G, feed_dict={self.z: batch_z, self.y: batch_labels})

                    else:
                        for batch_num in range(self.teachers_batch):
                            if self.teachers_batch > 1:
                                could_load, checkpoint_counter = self.load_d(self.teacher_dir, epoch=epoch,
                                                                             batch_num=batch_num)
                                if could_load:
                                    counter = checkpoint_counter
                                    print("load sucess")
                                else:
                                    print('fail')

                            for teacher_id in range(self.batch_teachers):
                                img_grads = self.sess.run(self.teachers_list[teacher_id]['img_grads'],
                                                          feed_dict={
                                                              self.z: batch_z,
                                                          })
                                img_grads_list.append(img_grads)

                        old_img = self.sess.run(self.G, feed_dict={self.z: batch_z})

                    img_grads_agg_list = []
                    for j in range(self.batch_size):
                        thresh = self.thresh

                        if config.non_private:
                            img_grads_agg_tmp = self.non_private_aggregation([grads[j] for grads in img_grads_list],
                                                                             config)
                            rdp_budget = 0
                        elif config.increasing_dim:
                            img_grads_agg_tmp, rdp_budget = self.aggregate_results(
                                [grads[j] for grads in img_grads_list], config, thresh=thresh, epoch=epoch)
                        elif config.signsgd_dept:
                            img_grads_agg_tmp, rdp_budget, rdp_budget_dept = self.aggregate_results(
                                [grads[j] for grads in img_grads_list], config, thresh=thresh)
                            self.rdp_counter_dept += rdp_budget_dept
                        else:
                            img_grads_agg_tmp, rdp_budget = self.aggregate_results(
                                [grads[j] for grads in img_grads_list], config, thresh=thresh)

                        img_grads_agg_list.append(img_grads_agg_tmp)
                        self.rdp_counter += rdp_budget

                    img_grads_agg = np.asarray(img_grads_agg_list)
                    updated_img = old_img + img_grads_agg

                    if config.non_private:
                        eps = 0
                        order = 0
                    else:
                        # calculate privacy budget and break if exceeds threshold
                        eps, order = compute_eps_from_delta(self.orders, self.rdp_counter, self.dp_delta)
                        if config.signsgd_dept:
                            eps_dept, order_dept = compute_eps_from_delta(self.orders, self.rdp_counter_dept, self.dp_delta)

                        if eps > config.max_eps:
                            print("New budget (eps = %.2f) exceeds threshold of %.2f. Early break (eps = %.2f)." % (
                                eps, config.max_eps, self.dp_eps_list[-1]))

                            # save privacy budget
                            self.save(config.checkpoint_dir, counter)
                            np.savetxt(self.checkpoint_dir + "/dp_eps.txt", np.asarray(self.dp_eps_list), delimiter=",")
                            np.savetxt(self.checkpoint_dir + "/rdp_eps.txt", np.asarray(self.rdp_eps_list),
                                       delimiter=",")
                            np.savetxt(self.checkpoint_dir + "/rdp_order.txt", np.asarray(self.rdp_order_list),
                                       delimiter=",")
                            if config.signsgd_dept:
                                np.savetxt(self.checkpoint_dir + "/dept_dp_eps.txt", np.asarray(self.dp_eps_list_dept),
                                           delimiter=",")
                                np.savetxt(self.checkpoint_dir + "/dept_rdp_eps.txt", np.asarray(self.rdp_eps_list_dept),
                                           delimiter=",")
                                np.savetxt(self.checkpoint_dir + "/dept_rdp_order.txt", np.asarray(self.rdp_order_list_dept),
                                           delimiter=",")

                            gen_batch = 100000 // self.batch_size + 1
                            data = self.gen_data(gen_batch)
                            data = data[:100000]
                            import joblib
                            interval = 100000 // 10
                            for slice in range(10):
                                joblib.dump(data[slice * interval: (slice+1) * interval], self.checkpoint_dir + '/eps-%.2f.data' % self.dp_eps_list[-1] + f'-{slice}.pkl')
                            sys.exit()

                    self.dp_eps_list.append(eps)
                    self.rdp_order_list.append(order)
                    self.rdp_eps_list.append(self.rdp_counter)
                    if config.signsgd_dept:
                        self.dp_eps_list_dept.append(eps_dept)
                        self.rdp_order_list_dept.append(order_dept)
                        self.rdp_eps_list_dept.append(self.rdp_counter_dept)

                    # Update G network
                    if self.y is not None:
                        _, summary_str, errG2 = self.sess.run([g_optim, self.g_sum, self.g_loss],
                                                              feed_dict={
                                                                  self.z: batch_z,
                                                                  self.updated_img: updated_img,
                                                                  self.y: batch_labels,
                                                              })
                        self.writer.add_summary(summary_str, epoch)

                        errG = self.g_loss.eval({
                            self.z: batch_z,
                            self.updated_img: updated_img,
                            self.y: batch_labels,
                        })
                    else:
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                       feed_dict={
                                                           self.z: batch_z,
                                                           self.updated_img: updated_img,
                                                       })
                        self.writer.add_summary(summary_str, epoch)

                        errG = self.g_loss.eval({
                            self.z: batch_z,
                            self.updated_img: updated_img,
                        })

                counter += 1
                print(
                    "Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, g_loss_before: %.8f, dp_eps: %.8f, rdp_order: %d" \
                    % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errD, errG, errG2, eps, order))
            # filename = 'epoch'+str(epoch)+'_errD'+str(errD)+'_errG'+str(errG)+'_teachers'+str(self.batch_teachers)+'f.csv'
            # if epoch % 4 == 0:
            print('----------------------generate sample----------------------')
            # data = self.gen_data(500)
            # output_dir = os.path.join(self.checkpoint_dir, self.sample_dir)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # filename = 'private.data_epoch_' + str(epoch) + '.pkl'
            # outfile = os.path.join(output_dir, filename)
            # mkdir(output_dir)
            # with open(outfile,'wb') as f:
            #     pickle.dump(data, f)

            filename = 'epoch' + str(epoch) + '_errD' + str(errD) + '_errG' + str(errG) + '_teachers' + str(
                self.batch_teachers) + 'f.csv'

            # save each epoch
            self.save(config.checkpoint_dir, counter)
            np.savetxt(self.checkpoint_dir + "/dp_eps.txt", np.asarray(self.dp_eps_list), delimiter=",")
            np.savetxt(self.checkpoint_dir + "/rdp_order.txt", np.asarray(self.rdp_order_list), delimiter=",")
            np.savetxt(self.checkpoint_dir + "/rdp_eps.txt", np.asarray(self.rdp_eps_list), delimiter=",")
            if config.signsgd_dept:
                np.savetxt(self.checkpoint_dir + "/dept_dp_eps.txt", np.asarray(self.dp_eps_list_dept),
                           delimiter=",")
                np.savetxt(self.checkpoint_dir + "/dept_rdp_eps.txt", np.asarray(self.rdp_eps_list_dept),
                           delimiter=",")
                np.savetxt(self.checkpoint_dir + "/dept_rdp_order.txt", np.asarray(self.rdp_order_list_dept),
                           delimiter=",")

            if config.save_epoch:
                floor_eps = math.floor(eps * 10) / 10.0
                if not self.save_dict[floor_eps]:
                    # get a checkpoint of low eps
                    self.save_dict[floor_eps] = True
                    from shutil import copytree
                    src_dir = os.path.join(config.checkpoint_dir, self.model_dir)
                    dst_dir = os.path.join(config.checkpoint_dir, str(floor_eps))
                    copytree(src_dir, dst_dir)

        #
        # save after training
        self.save(config.checkpoint_dir, counter)
        np.savetxt(self.checkpoint_dir + "/dp_eps.txt", np.asarray(self.dp_eps_list), delimiter=",")
        np.savetxt(self.checkpoint_dir + "/rdp_eps.txt", np.asarray(self.rdp_eps_list), delimiter=",")
        np.savetxt(self.checkpoint_dir + "/rdp_order.txt", np.asarray(self.rdp_order_list), delimiter=",")
        if config.signsgd_dept:
            np.savetxt(self.checkpoint_dir + "/dept_dp_eps.txt", np.asarray(self.dp_eps_list_dept),
                       delimiter=",")
            np.savetxt(self.checkpoint_dir + "/dept_rdp_eps.txt", np.asarray(self.rdp_eps_list_dept),
                       delimiter=",")
            np.savetxt(self.checkpoint_dir + "/dept_rdp_order.txt", np.asarray(self.rdp_order_list_dept),
                       delimiter=",")

        return self.dp_eps_list[-1], self.dp_delta

    def discriminator(self, image, y):
        if self.config.simple_gan:
            return self.simple_discriminator(image, y)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        if self.wgan:
            h1 = lrelu(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv'))
        else:
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))

        h1 = tf.reshape(h1, [self.batch_size, -1])
        h1 = concat([h1, y], 1)

        if self.wgan:
            h2 = lrelu(linear(h1, self.dfc_dim, 'd_h2_lin'))
        else:
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3

    def simple_discriminator(self, image, y):
        # yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        image = tf.reshape(image, (-1, 20))
        x = concat([image, y], 1)

        h0 = tf.nn.relu(linear(x, 100, 'd_h0_lin'))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(linear(h0, 50, 'd_h1_lin'))
        h1 = concat([h1, y], 1)

        h2 = linear(h1, 1, 'd_h2_lin')

        return tf.nn.sigmoid(h2), h2

    def simple_generator(self, z, y):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            h0 = tf.nn.relu(linear(z, 200, 'g_h0_lin'))
            h0 = concat([h0, y], 1)

            h1 = tf.nn.relu(linear(h0, 50, 'g_h1_lin'))
            h1 = concat([h1, y], 1)

            h2 = linear(h1, 20, 'g_h2_lin')
            return tf.reshape(h2, (-1, 2, 2, 5))

    def generator(self, z, y):
        if self.config.simple_gan:
            return self.simple_generator(z, y)
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h / 2), max(int(s_h / 4), 1)
            s_w2, s_w4 = int(s_w / 2), max(int(s_w / 4), 1)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            if self.wgan:
                h0 = tf.nn.relu(linear(z, self.gfc_dim, 'g_h0_lin'))
            else:
                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = concat([h0, y], 1)

            if self.wgan:
                h1 = tf.nn.relu(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'))
            else:
                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            if self.wgan:
                h2 = tf.nn.relu(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'))
            else:
                h2 = tf.nn.relu(
                    self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            # if 'ae' in self.config.dataset:
            #     return deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')

            if self.config.tanh:
                return (1 + tf.nn.tanh(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))) / 2.
            else:
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def gen_data(self, n_batch, label=None):
        output_list = []
        for i in range(n_batch):
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            if self.y is not None:
                if label is None:
                    batch_labels = self.get_random_labels(self.batch_size)
                else:
                    batch_labels = np.zeros((self.batch_size, self.y_dim), dtype=np.float)
                    batch_labels[:, label] = 1.0

                outputs = self.sess.run(self.G,
                                        feed_dict={
                                            self.z: batch_z,
                                            self.y: batch_labels,
                                        })
                outputsX = outputs.reshape([self.batch_size, -1])
                outputs = np.hstack([outputsX, batch_labels[:, 0:10]])
            else:
                outputs = self.sess.run(self.G,
                                        feed_dict={
                                            self.z: batch_z,
                                        })
                outputsX = outputs.reshape([self.batch_size, -1])
                outputs = outputsX

            output_list.append(outputs)

        output_arr = np.vstack(output_list)
        return output_arr

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def print_tensors_in_checkpoint(self, checkpoint_dir, ckpt_name):
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        import os
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
        # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
        print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)

    def load_pretrain(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        print(checkpoint_dir)
        save_vars_dict = {x.name[:-2]: x for x in self.save_vars if x.name.startswith('generator')}
        pretrain_saver = tf.train.Saver(max_to_keep=5, var_list=save_vars_dict)
        print(self.dataset_name)
        if 'cifar' in self.dataset_name or 'cinic' in self.dataset_name:
            ckpt_name = 'DCGAN.model-100'
        elif 'mnist' in self.dataset_name:
            ckpt_name = 'CIFAR.model-250'
        elif 'celebA' in self.dataset_name:
            ckpt_name = 'CIFAR.model-99'
        pretrain_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        import re
        if self.config.load_d:
            for i in range(self.batch_teachers):
                print('loading teacher {}'.format(i))
                save_vars_dict = {re.sub(r'teacher[0-9]+', 'teacher0', x.name[:-2]): x for x in self.save_vars if
                                  x.name.startswith('teacher{}/'.format(i))}
                pretrain_saver = tf.train.Saver(max_to_keep=5, var_list=save_vars_dict)
                pretrain_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

        # save_vars_dict = {x.name: x for x in self.save_vars}
        # print(save_vars_dict.keys())
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        # current_scope = tf.contrib.framework.get_name_scope()
        # with tf.variable_scope(current_scope, reuse=True):
        #     biases = tf.get_variable("teacher0/d_h0_conv/biases")
        #     biases2 = tf.get_variable("teacher12/d_h0_conv/biases")
        #     biases3 = tf.get_variable("generator/g_h0_lin/Matrix")
        #     biases = tf.Print(biases, [biases, biases2, biases3])
        #     self.sess.run(biases)
        return True, counter

    def load(self, checkpoint_dir, ckpt_name):
        import re
        print(" [*] Reading checkpoints...")
        print(checkpoint_dir)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter

    # def load(self, checkpoint_dir):
    #     import re
    #     print(" [*] Reading checkpoints...")
    #     checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    #     print(checkpoint_dir)
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     print(ckpt)
    #     print(ckpt.model_checkpoint_path)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #         print(ckpt_name)
    #         self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #         counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
    #         print(" [*] Success to read {}".format(ckpt_name))
    #         return True, counter
    #     else:
    #         print(" [*] Failed to find a checkpoint")
    #         return False, 0

    def load_d(self, checkpoint_dir, batch_num, epoch):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        model_name = "DCGAN_batch_" + str(batch_num) + "_epoch-" + str(epoch)

        ckpt = os.path.join(checkpoint_dir, model_name)
        print(ckpt + ".meta")
        if os.path.isfile(ckpt + ".meta"):
            # model_name = "DCGAN_batch_" + str(batch_num) + "_epoch_" + str(epoch)
            # print(model_name)
            self.saver_d.restore(self.sess, ckpt)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", model_name)).group(0))
            print(" [*] Success to read {}".format(model_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, checkpoint_dir, step):
        model_name = "CIFAR.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def save_d(self, checkpoint_dir, step, teacher_batch):
        model_name = "DCGAN_batch_" + str(teacher_batch) + "_epoch"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver_d.save(self.sess,
                          os.path.join(checkpoint_dir, model_name),
                          global_step=step)
        print("-------------save-dis----------------------")

    def save_g(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver_g.save(self.sess,
                          os.path.join(checkpoint_dir, model_name),
                          global_step=step)
