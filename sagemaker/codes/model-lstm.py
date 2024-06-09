#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import warnings
import glob
import json

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import LSTM

import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

## Required parameters
tf.compat.v1.flags.DEFINE_string(
    "activation", 'relu',
    "activation function definition")

tf.compat.v1.flags.DEFINE_integer(
    "cell_num", 10,
    "Cell number")

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

    X_train_lmse = np.load(FLAGS.data_dir + '/train/' + 'X_train.npy')
    y_train = np.load(FLAGS.data_dir + '/train/' + 'y_train.npy')

    print('Processed data loaded!')

    CKPT_FILE_PATH = FLAGS.checkpoint_dir
    if not os.path.isdir(CKPT_FILE_PATH):
        os.mkdir(CKPT_FILE_PATH)

    checkpoint_path = CKPT_FILE_PATH + "/cp.ckpt"

    lstm_model = Sequential()
    lstm_model.add(LSTM(FLAGS.cell_num, input_shape=(1, X_train_lmse.shape[1]), activation=FLAGS.activation,
                        kernel_initializer='lecun_uniform', return_sequences=False))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss=FLAGS.loss, optimizer=FLAGS.optimizer)

    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=False,
                                  verbose=1)

    lstm_model.fit(X_train_lmse, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, verbose=1, shuffle=False,
                   callbacks=[cp_callback])
    lstm_model.save(FLAGS.output_dir + '/model.h5')

    print('LSTM Model training finished!')
