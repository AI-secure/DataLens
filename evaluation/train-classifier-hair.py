#!/usr/bin/env python
# coding: utf-8

# In[13]:
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train classifier and evaluate their accuracy')
parser.add_argument('--data', type=str, help='datafile name')
parser.add_argument('--range', type=int, default=10, help="range of pickles")

args = parser.parse_args()

data = np.zeros((100000, 12291))
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
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.keras.backend.set_session(tf.Session(config=config));


# In[5]:


def load_celeb():
    celebA_directory = '../../data/celebA/'
    tst_x = joblib.load(celebA_directory + 'celeb-tst-ups-hair-x.pkl')
    tst_y = joblib.load(celebA_directory + 'celeb-tst-ups-hair-y.pkl')
    print(tst_y.sum(), len(tst_y))
    from keras.utils import np_utils
    tst_y = np_utils.to_categorical(tst_y, 3)
    return tst_x, tst_y


x_test, y_test = load_celeb()



def pipeline(data):
    print(data.shape)
    x, label = np.hsplit(data, [-3])
    nb_classes = 3
    label = label.reshape((label.shape[0], nb_classes),order='F')
    x = x.reshape(x.shape[0], 64, 64, 3)

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.convolutional import Convolution2D, Conv2D
    from keras.optimizers import Adam
    from keras import optimizers


    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(64, 64, 3), name='Conv2D-1'))
    model.add(MaxPooling2D(pool_size=2, name='MaxPool'))
    model.add(Dropout(0.2, name='Dropout-1'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'))
    model.add(Dropout(0.25, name='Dropout-2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='Dense'))
    model.add(Dense(nb_classes, activation='softmax', name='Output'))
    sgd = optimizers.sgd(lr=1e-4) #, decay=1e-6, momentum=0.9, nesterov=True)


    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    print(x.shape)
    print(label.shape)
    print(x_test.shape)
    print(y_test.shape)
    evals = model.fit(x, label, batch_size=256, epochs=250, validation_data=(x_test, y_test), shuffle=True)
    return evals.history



hist = pipeline(data)
print("Max acc:", max(hist['val_acc']))

