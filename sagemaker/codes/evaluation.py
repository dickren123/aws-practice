#!/usr/bin/env python
# coding=utf-8

import argparse
import os

os.system('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple boto3 tensorflow==2.1.0 h5py==2.10.0')
import warnings
import glob
import json
import tarfile
import boto3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn import preprocessing
import tensorflow as tf

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sns-topic', type=str,
                        default='arn:aws-cn:sns:cn-northwest-1:542319707026:sm-airflow-evaluation')
    args, _ = parser.parse_known_args()

    print('Received arguments5 {}'.format(args))

    # 解析当前集群的机器分布情况
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found: default to one instance")
        pass  # Ignore

    print('resconfig :' + str(resconfig))

    input_dataset1 = '/opt/ml/processing/input-dataset1'
    input_dataset2 = '/opt/ml/processing/input-dataset2'
    input_model1 = '/opt/ml/processing/input-model1'
    input_model2 = '/opt/ml/processing/input-model2'

    X_test = np.load(input_dataset1 + '/X_test.npy')
    y_test = np.load(input_dataset1 + '/y_test.npy')

    X_test_lmse = np.load(input_dataset2 + '/X_test.npy')
    y_test_lmse = np.load(input_dataset2 + '/y_test.npy')

    untar(input_model1 + "/model.tar.gz", input_model1)
    untar(input_model2 + "/model.tar.gz", input_model2)

    nn_model = tf.keras.models.load_model(input_model1 + '/model.h5')
    lstm_model = tf.keras.models.load_model(input_model2 + '/model.h5')

    nn_test_mse = nn_model.evaluate(X_test, y_test, batch_size=1)
    print('NN: %f' % nn_test_mse)
    lstm_test_mse = lstm_model.evaluate(X_test_lmse, y_test_lmse, batch_size=1)
    print('LSTM: %f' % lstm_test_mse)

    with open("/opt/ml/processing/output/test-output.txt", "w", encoding='utf-8') as f:
        f.write('NN    {}\nLSTM    {}'.format(nn_test_mse, lstm_test_mse))

    sns_message = 'Training Workflow finishes, NN Model evaluation MSE is {}, LSTM Model evaluation MSE is {}'.format(
        nn_test_mse, lstm_test_mse)

    # trigger sns
    sns = boto3.client('sns', region_name='cn-northwest-1')
    response = sns.publish(
        TopicArn=args.sns_topic,
        Message=sns_message
    )

