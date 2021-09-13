import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables, mkdir

import tensorflow as tf
import argparse
from gen_data import batch2str
import sys
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch for training teacher models")
flags.DEFINE_integer("g_epoch", 500, "Epoch for training the student models")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 30, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 32, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 32,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 32,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "slt", "The name of dataset [cinic, celebA, mnist, lsun, fire-small]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint_name", "checkpoint", "checkpoint model name [checkpoint]")

flags.DEFINE_string("data_dir", "../../data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("pretrain", True, "True for loading the pretrained models, False for not load [True]")
flags.DEFINE_boolean("load_d", True,
                     "True for loading the pretrained models w/ discriminator, False for not load [True]")
flags.DEFINE_boolean("crop", False, "True for cropping")
flags.DEFINE_integer("orders", 200, "rdp orders")
flags.DEFINE_integer("proj_mat", 1, "#/ projection mat")
flags.DEFINE_integer("z_dim", 100, "#/ z dim")
flags.DEFINE_integer("y_dim", 10, "#/ y dim")
flags.DEFINE_boolean("tanh", False, "Use tanh as activation func")

flags.DEFINE_boolean("random_proj", True, "Apply pca for gradient aggregation ")
flags.DEFINE_boolean("simple_gan", False, "Use fc to build GAN")
flags.DEFINE_boolean("mean_kernel", False, "Apply Mean Kernel for gradient agggregation")
flags.DEFINE_boolean("signsgd", False, "Apply sign sgd for gradient agggregation")
flags.DEFINE_boolean("signsgd_nothresh", False, "Apply sign sgd for gradient agggregation")
flags.DEFINE_boolean("klevelsgd", False, "Apply klevel sgd for gradient agggregation")
flags.DEFINE_boolean("sketchsgd", False, "Apply sketch sgd for gradient agggregation")
flags.DEFINE_boolean("signsgd_dept", False, "Apply sign sgd for gradient agggregation with data dependent bound")
flags.DEFINE_boolean("stochastic", False, "Apply stochastic sign sgd for gradient agggregation")
flags.DEFINE_integer("pretrain_teacher", 0, "Pretrain teacher for epochs")
flags.DEFINE_boolean("save_vote", False, "Save voting results")
flags.DEFINE_boolean("pca", False, "Apply pca for gradient aggregation ")
flags.DEFINE_boolean("non_private", False, "Do not apply differential privacy")
flags.DEFINE_boolean("increasing_dim", False, "Increase the projection dimension for each epoch")
flags.DEFINE_boolean("wgan", False, "Train wgan")
flags.DEFINE_boolean("small", False, "Use a smaller discriminator")
flags.DEFINE_float("sigma", 2000.0, "Scale of gaussian noise for gradient aggregation")
flags.DEFINE_float("sigma_thresh", 4500.0, "Scale of gaussian noise for thresh gnmax")
flags.DEFINE_float("pca_sigma", 1.0, "Scale of gaussian noise for dp pca")
flags.DEFINE_float("step_size", 1e-4, "Step size for gradient aggregation")
flags.DEFINE_float("delta", 1e-5, "delta for differential privacy")
flags.DEFINE_integer("g_step", 1, "steps of the generator")
flags.DEFINE_integer("d_step", 1, "steps of the discriminator")
flags.DEFINE_integer("pca_dim", 10, "principal dimensions for pca")
flags.DEFINE_float("thresh", 0.5, "threshhold for threshgmax")
flags.DEFINE_float("max_eps", 1, "maximum epsilon")
flags.DEFINE_float("max_grad", 0, "maximum gradient for signsgd aggregation")
flags.DEFINE_boolean("random_label", False, "random labels for training data, only used when pretraining some models")
flags.DEFINE_boolean("shuffle", True, "Evenly distribute dataset")
flags.DEFINE_boolean("save_epoch", False, "Save each epoch per 0.1 eps")
flags.DEFINE_integer("batch_teachers", 1, "Number of teacher models in one batch")
flags.DEFINE_integer("teachers_batch", 1, "Number of batch")
flags.DEFINE_integer("topk", 50, "Number of top k gradients")
flags.DEFINE_integer("klevel", 4, "Levels of gradient quantization")
flags.DEFINE_string("teacher_dir", "teacher", "Directory name to save the teacher [teacher]")
flags.DEFINE_string("generator_dir", "generator", "Directory name to save the generator")
flags.DEFINE_string("loss", "l1", "AE reconstruction loss")
flags.DEFINE_string("ae", "", "AE model name")
flags.DEFINE_boolean("train_ae", False, "Train ae")
flags.DEFINE_boolean("finetune_ae", False, "Finetune ae")
flags.DEFINE_integer("sample_step", 10, "Number of teacher models in one batch")
flags.DEFINE_integer("hid_dim", 512, "Dimmension of hidden dim")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.flag_values_dict())

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    if FLAGS.thresh == 0:
        thresh = None
    else:
        thresh = FLAGS.thresh

    if FLAGS.wgan:
        FLAGS.learning_rate = 5e-5
        FLAGS.step_size = 5e-4

    with tf.Session(config=run_config) as sess:

        dcgan = DCGAN(
            sess,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            y_dim=FLAGS.y_dim,
            z_dim=FLAGS.z_dim,
            dataset_name=FLAGS.dataset,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            data_dir=FLAGS.data_dir,
            # parameters to tune
            batch_teachers=FLAGS.batch_teachers,
            pca=FLAGS.pca,
            random_proj=FLAGS.random_proj,
            thresh=thresh,
            dp_delta=FLAGS.delta,
            pca_dim=FLAGS.pca_dim,
            teachers_batch=FLAGS.teachers_batch,
            teacher_dir=os.path.join(FLAGS.checkpoint_dir, FLAGS.teacher_dir),
            generator_dir=FLAGS.generator_dir,
            non_private=FLAGS.non_private,
            input_height=FLAGS.input_height,
            input_width=FLAGS.input_width,
            output_height=FLAGS.output_height,
            output_width=FLAGS.output_width,
            wgan=FLAGS.wgan,
            small=FLAGS.small,
            config=FLAGS
        )

        show_all_variables()

        if FLAGS.train_ae and FLAGS.ae:
            dcgan.train_ae()
        elif FLAGS.finetune_ae and FLAGS.ae:
            dcgan.finetune_ae()
        else:
            if FLAGS.train:
                if FLAGS.ae:
                    pass
                else:
                    epsilon, delta = dcgan.train_together(FLAGS)
                    filename = '%.2fepsilon-%.2fdelta.data' % (epsilon, delta)
            else:
                if not dcgan.load(FLAGS.checkpoint_dir, FLAGS.checkpoint_name)[0]:
                    raise Exception("[!] Train a model first, then run test mode")
                filename = 'private.data'

            outpath = os.path.join(FLAGS.checkpoint_dir, FLAGS.sample_dir)
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            outfile = os.path.join(outpath, filename)
            n_batch = 100000 // FLAGS.batch_size + 1
            data = dcgan.gen_data(n_batch)
            data = data[:100000]
            import joblib

            joblib.dump(data, outfile)


if __name__ == '__main__':
    tf.app.run()
