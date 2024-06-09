import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor

region = boto3.session.Session().region_name
role = 'arn:aws-cn:iam::542319707026:role/ec2s3'

sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                     role=role,
                                     instance_type='ml.m5.xlarge',
                                     instance_count=1)

bucket = sagemaker.Session().default_bucket()
output_prefix = 'jiuzhang/evaluation-result'

input_prefix = 'jiuzhang/raw-data-2020-11-10'
input_preprocessed_prefix = 'jiuzhang/processed-data-2020-11-10'
prefix = 'jiuzhang'

input_data = 's3://{}/{}/'.format(bucket, input_prefix)
output_nn_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'nn-train')
output_nn_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'nn-test')
output_lstm_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'lstm-train')
output_lstm_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'lstm-test')

input_model1 = 's3://sagemaker-cn-northwest-1-542319707026/jiuzhang/training-output-NN/group-a-xiaoming-nn-training-2020-11-06-13-27-49-248/output/'
input_model2 = 's3://sagemaker-cn-northwest-1-542319707026/jiuzhang/training-output-LSTM/group-a-xiaoming-lstm-training-2020-11-06-13-28-04-466/output/'

input_nn_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'nn-test')
input_lstm_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'lstm-test')

output_evaluatioon_result = 's3://{}/{}/'.format(bucket, output_prefix)

from sagemaker.processing import ProcessingInput, ProcessingOutput

_inputs = [ProcessingInput(source=input_model1,  # 指定s3的原始数据路径
                           destination='/opt/ml/processing/input-model1',
                           s3_data_distribution_type='ShardedByS3Key'),
           ProcessingInput(source=input_model2,  # 指定s3的原始数据路径
                           destination='/opt/ml/processing/input-model2',
                           s3_data_distribution_type='ShardedByS3Key'),
           ProcessingInput(source=input_nn_test_data,  # 指定s3的原始数据路径
                           destination='/opt/ml/processing/input-dataset1',
                           s3_data_distribution_type='ShardedByS3Key'),
           ProcessingInput(source=input_lstm_test_data,  # 指定s3的原始数据路径
                           destination='/opt/ml/processing/input-dataset2',
                           s3_data_distribution_type='ShardedByS3Key'),
           ]

_outputs = [ProcessingOutput(output_name='evaluatioin_result',
                             source='/opt/ml/processing/output',
                             destination=output_evaluatioon_result)]

sklearn_processor.run(code='../codes/evaluation.py',  # 自带的处理代码
                      inputs=_inputs,
                      outputs=_outputs,
                      wait=False,
                      arguments=['--sns-topic', 'arn:aws-cn:sns:cn-northwest-1:542319707026:sm-airflow-evaluation']
                      )

preprocessing_job_description = sklearn_processor.jobs[-1].describe()