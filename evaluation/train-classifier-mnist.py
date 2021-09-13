#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train classifier and evaluate their accuracy')
parser.add_argument('--data', type=str, help='datafile name')
parser.add_argument('--range', type=int, default=10, help="range of pickles")

args = parser.parse_args()

data = np.zeros((100000, 794))
dim = 0
import joblib
from tqdm import tqdm
for i in tqdm(range(args.range)):
    x =  joblib.load(args.data + f'-{i}.pkl')
    data[dim: dim+len(x)] = x
    dim += len(x)
print(args.data)
print(data.shape)


import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config));


def pipeline(data):
    x, label = np.hsplit(data, [-10])
    nb_classes = 10
    label = label.reshape((label.shape[0], nb_classes), order='F')
    x = x.reshape(x.shape[0], 28, 28, 1)
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32') / 255.

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.convolutional import Convolution2D, Conv2D
    from keras.optimizers import Adam
    from keras import optimizers

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), name='Conv2D-1'))
    model.add(MaxPooling2D(pool_size=2, name='MaxPool'))
    model.add(Dropout(0.2, name='Dropout-1'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'))
    model.add(Dropout(0.25, name='Dropout-2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='Dense'))
    model.add(Dense(nb_classes, activation='softmax', name='Output'))
    sgd = optimizers.sgd(lr=2e-3)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    print(x.shape)
    print(label.shape)
    print(x_test.shape)
    print(y_test.shape)
    train_accs = []
    eval_accs = []
    history = model.fit(x, label, batch_size=512, epochs=600, validation_data=(x_test, y_test), shuffle=True)
    if 'acc' in history.history:
        train_accs = history.history['acc']
    else:
        train_accs = history.history['accuracy']
    if 'val_acc' in history.history:
        eval_accs = history.history['val_acc']
    else:
        eval_accs = history.history['val_accuracy']
    return train_accs, eval_accs




train_accs, eval_accs = pipeline(data)
print("Max eval acc:", max(eval_accs))
print("Max train acc:", max(train_accs))

