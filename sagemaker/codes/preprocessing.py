#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import warnings
import glob
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn import preprocessing

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-date', type=str, default='2018-01-01')
    args, _ = parser.parse_known_args()

    print('Received argument {}'.format(args))

    # 解析当前集群的机器分布情况
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found: default to one instance")
        pass  # Ignore

    print('resconfig :' + str(resconfig))

    input_data_path = '/opt/ml/processing/input/'
    print('Reading input data6 from {}'.format(input_data_path))

    print('now_files: ' + str(os.listdir(input_data_path)))

    _filenames = [i for i in glob.glob(input_data_path + '*.{}'.format('csv'))]
    df = pd.concat([pd.read_csv(f, index_col=None, header=0) for f in _filenames], axis=0, ignore_index=True)

    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date'], drop=True)

    # split dataset
    split_date = args.split_date
    df = df['Adj Close']
    train = df.loc[:split_date]
    test = df.loc[split_date:]

    # scale train and test data to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train.to_numpy().reshape(-1, 1))
    test_sc = scaler.transform(test.to_numpy().reshape(-1, 1))

    X_train = train_sc[:-1]
    y_train = train_sc[1:]

    X_test = test_sc[:-1]
    y_test = test_sc[1:]

    # save dataset
    np.save('/opt/ml/processing/nn-train/X_train', X_train)
    np.save('/opt/ml/processing/nn-train/y_train', y_train)

    np.save('/opt/ml/processing/nn-test/X_test', X_test)
    np.save('/opt/ml/processing/nn-test/y_test', y_test)

    ## below is for lstm dataset
    train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
    test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)

    for s in range(1, 2):
        train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
        test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)

    X_train = train_sc_df.dropna().drop('Y', axis=1)
    y_train = train_sc_df.dropna().drop('X_1', axis=1)

    X_test = test_sc_df.dropna().drop('Y', axis=1)
    y_test = test_sc_df.dropna().drop('X_1', axis=1)

    X_train = X_train.values
    y_train = y_train.values

    X_test = X_test.values
    y_test = y_test.values

    X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print('LSTM Train shape: ', X_train_lmse.shape)
    print('LSTM Test shape: ', X_test_lmse.shape)

    np.save('/opt/ml/processing/lstm-train/X_train', X_train_lmse)
    np.save('/opt/ml/processing/lstm-train/y_train', y_train)

    np.save('/opt/ml/processing/lstm-test/X_test', X_test_lmse)
    np.save('/opt/ml/processing/lstm-test/y_test', y_test)

    print('Splitting data into train and test sets with split_date {}'.format(split_date))
    print('Processed Dataset has been uploaded to {}'.format('jiuzhang/processed-data-2020-11-10'))