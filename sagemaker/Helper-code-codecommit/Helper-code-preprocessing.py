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
input_prefix = 'jiuzhang/raw-data-2020-11-10'
input_preprocessed_prefix = 'jiuzhang/processed-data-2020-11-10'
prefix = 'jiuzhang'

input_data = 's3://{}/{}/'.format(bucket, input_prefix)
output_nn_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'nn-train')
output_nn_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'nn-test')
output_lstm_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'lstm-train')
output_lstm_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix,'lstm-test')

from sagemaker.processing import ProcessingInput, ProcessingOutput

_inputs = [ProcessingInput(source=input_data,  # 指定s3的原始数据路径
                           destination='/opt/ml/processing/input',
                           s3_data_distribution_type='ShardedByS3Key')]

_outputs = [ProcessingOutput(output_name='nn_train_data',
                             source='/opt/ml/processing/nn-train',
                             destination=output_nn_train_data),
            ProcessingOutput(output_name='nn_test_data',
                             source='/opt/ml/processing/nn-test',
                             destination=output_nn_test_data),
            ProcessingOutput(output_name='lstm_train_data',
                             source='/opt/ml/processing/lstm-train',
                             destination=output_lstm_train_data),
            ProcessingOutput(output_name='lstm_test_data',
                             source='/opt/ml/processing/lstm-test',
                             destination=output_lstm_test_data)]

sklearn_processor.run(code='s3://sagemaker-cn-northwest-1-542319707026/sagemaker-repo/codes/preprocessing.py',  # 自带的处理代码
                      inputs=_inputs,
                      outputs=_outputs,
                      wait=False,
                      arguments=['--split-date', '2018-01-01']
                      )

preprocessing_job_description = sklearn_processor.jobs[-1].describe()

output_config = preprocessing_job_description['ProcessingOutputConfig']
for output in output_config['Outputs']:
    if output['OutputName'] == 'train_data':
        preprocessed_training_data = output['S3Output']['S3Uri']
    if output['OutputName'] == 'test_data':
        preprocessed_test_data = output['S3Output']['S3Uri']