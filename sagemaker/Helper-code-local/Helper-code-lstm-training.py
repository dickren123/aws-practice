import sagemaker
import os
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

import sagemaker
import os
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

bucket = sagemaker.Session().default_bucket()
role = 'arn:aws-cn:iam::542319707026:role/ec2s3'

input_prefix = 'jiuzhang/raw-data-2020-11-10'
input_preprocessed_prefix = 'jiuzhang/processed-data-2020-11-10'
prefix = 'jiuzhang'

_output_path = 's3://{}/{}/training-output-LSTM'.format(bucket,prefix)

input_data = 's3://{}/{}/'.format(bucket, input_prefix)
output_nn_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'nn-train')
output_nn_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'nn-test')
output_lstm_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'lstm-train')
output_lstm_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'lstm-test')

_hyperparameters = {'activation': 'relu',
                   'cell_num': 12,
                   'loss': 'mean_squared_error',
                   'data_dir': '/opt/ml/input/data',
                   'output_dir': '/opt/ml/model',
                   'checkpoint_dir': '/opt/ml/checkpoints',
                   'optimizer': 'adam',
                   'epochs': 10,
                   'batch_size': 1}

estimator2 = TensorFlow(entry_point='model-lstm.py',
                       source_dir='../codes',
                       base_job_name = 'group-a-xiaoming-lstm-training',
                       instance_type='ml.p3.2xlarge', # V100 单卡
                       image_uri = '542319707026.dkr.ecr.cn-northwest-1.amazonaws.com.cn/group-a-xiaoming-lstm',
                       instance_count=1,
                       hyperparameters=_hyperparameters,
                       role=role,
                       metric_definitions = [{'Name': 'train:loss', 'Regex': '.*loss: ([0-9\\.]+)'}],
                       output_path = _output_path,
                       framework_version='2.1.0',
                       checkpoint_local_path = '/opt/ml/checkpoints/',
                       checkpoint_s3_uri = _output_path + '/checkpoints',
                       debugger_hook_config=False,
                       py_version='py3')

estimator2.fit({'train': output_lstm_train_data, 'test':output_lstm_test_data},wait=True)

