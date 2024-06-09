#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import warnings
import glob
import json

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM

import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

## Required parameters
tf.compat.v1.flags.DEFINE_string(
    "activation", 'relu',
    "activation function definition")

tf.compat.v1.flags.DEFINE_integer(
    "hidden_layer", 10,
    "Hidden layer number")

tf.compat.v1.flags.DEFINE_string("loss", "mean_squared_error", "loss function")

tf.compat.v1.flags.DEFINE_string("data_dir", '/opt/ml/input/data',
                                 "input data dir")

tf.compat.v1.flags.DEFINE_string("checkpoint_dir", '/opt/ml/checkpoints',
                                 "input data dir")

tf.compat.v1.flags.DEFINE_string("model_dir", '',
                                 "model s3 dir")

tf.compat.v1.flags.DEFINE_string(
    "output_dir", "/opt/ml/model",
    "The output directory where the model checkpoints will be written.")

tf.compat.v1.flags.DEFINE_string("optimizer", "adam", "optimizer type")

tf.compat.v1.flags.DEFINE_integer(
    "epochs", 10,
    "epochs number")

tf.compat.v1.flags.DEFINE_integer("batch_size", 1, "batch size")

if __name__ == '__main__':

    X_train = np.load(FLAGS.data_dir + '/train/' + 'X_train.npy')
    y_train = np.load(FLAGS.data_dir + '/train/' + 'y_train.npy')

    print('Processed data loaded!')

    CKPT_FILE_PATH = FLAGS.checkpoint_dir
    if not os.path.isdir(CKPT_FILE_PATH):
        os.mkdir(CKPT_FILE_PATH)

    checkpoint_path = CKPT_FILE_PATH

    nn_model = Sequential()
    nn_model.add(Dense(FLAGS.hidden_layer, input_dim=1, activation=FLAGS.activation))
    nn_model.add(Dense(1))
    nn_model.compile(loss=FLAGS.loss, optimizer=FLAGS.optimizer)

    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    cp_callback = ModelCheckpoint(filepath=checkpoint_path + '/',
                                  save_weights_only=False,
                                  verbose=1)

    nn_model.fit(X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, verbose=1, callbacks=[cp_callback],
                 shuffle=False)
    nn_model.save(FLAGS.output_dir + '/model.h5')

    print('NN Model training finished!')
