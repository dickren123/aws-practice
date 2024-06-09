import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from time import gmtime, strftime
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.airflow import training_config,model_config_from_estimator,deploy_config,deploy_config_from_estimator
from sagemaker.model import Model
from datetime import timedelta
import json

import airflow
from airflow import DAG
from airflow.models import TaskInstance
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.sagemaker_training_operator import SageMakerTrainingOperator
from airflow.contrib.operators.sagemaker_model_operator import SageMakerModelOperator
from airflow.contrib.operators.sagemaker_endpoint_operator import SageMakerEndpointOperator
from airflow import AirflowException
import airflow.utils

region = boto3.session.Session().region_name
role = get_execution_role()

bucket = sagemaker.Session().default_bucket()
input_prefix = 'jiuzhang/raw-data-2020-11-10'
input_preprocessed_prefix = 'jiuzhang/processed-data-2020-11-10'
prefix = 'jiuzhang'

input_data = 's3://{}/{}/'.format(bucket, input_prefix)
output_nn_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix, 'nn-train')
output_nn_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix, 'nn-test')
output_lstm_train_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix, 'lstm-train')
output_lstm_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix, 'lstm-test')

model_lstm_uri = ''
model_nn_uri = ''

role_arn = 'arn:aws-cn:iam::542319707026:role/service-role/AmazonSageMaker-ExecutionRole-20200403T105376'

model_lstm_training_image_uri = '542319707026.dkr.ecr.cn-northwest-1.amazonaws.com.cn/airflow-lstm'
model_inference_image_uri = '542319707026.dkr.ecr.cn-northwest-1.amazonaws.com.cn/byoc-infer-h5'

client_ssm = boto3.client('ssm')


default_args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
    'provide_context': True
}

dag = DAG('airflow_sagemaker_pipeline', default_args=default_args,
          schedule_interval='@once')

# set config timestamp
def config_timestamp(**context):
    context['task_instance'].xcom_push(key='timestamp', value=strftime("%Y-%m-%d-%H-%M-%S", gmtime()))


# preprocessing
def preprocessing(**context):
    _job_name = 'airflow-preprocessing-{}'.format(context['task_instance'].xcom_pull(key='timestamp'))

    from sagemaker.processing import ProcessingInput, ProcessingOutput

    sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                         role=role,
                                         instance_type='ml.m5.xlarge',
                                         instance_count=1)

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
                          job_name=_job_name,
                          arguments=['--split-date', '2018-01-01']
                          )

    return _job_name
# set git configuration
_git_config = {'repo': 'https://git-codecommit.cn-northwest-1.amazonaws.com.cn/v1/repos/sagemaker-repo.git',
              'branch': 'master'}

# estimator for NN model training
_output_path1 = 's3://{}/{}/training-output-NN'.format(bucket, prefix)

_hyperparameters1 = {'activation': 'relu',
                     'hidden_layer': 12,
                     'loss': 'mean_squared_error',
                     'data_dir': '/opt/ml/input/data',
                     'output_dir': '/opt/ml/model',
                     'checkpoint_dir': '/opt/ml/checkpoints',
                     'optimizer': 'adam',
                     'epochs': 50,
                     'batch_size': 1}

estimator_nn = TensorFlow(entry_point='model-nn.py',
                        source_dir='codes',
                        git_config=_git_config,
                        instance_type='ml.p3.2xlarge',  # V100
                        instance_count=1,
                        hyperparameters=_hyperparameters1,
                        role=role,
                        metric_definitions=[{'Name': 'train:loss', 'Regex': '.*loss: ([0-9\\.]+)'}],
                        output_path=_output_path1,
                        framework_version='2.1.0',
                        checkpoint_local_path='/opt/ml/checkpoints/',
                        checkpoint_s3_uri=_output_path1 + '/checkpoints',
                        debugger_hook_config=False,
                        py_version='py3')

data_input_dic1 = {'train': output_nn_train_data, 'test': output_nn_test_data}

# train_config specifies SageMaker training configuration for training operator
_train_config_nn = training_config(estimator=estimator_nn,
                                inputs=data_input_dic1,
                                job_name='airflow-nn-training-{}'.format(
                                    "{{ task_instance.xcom_pull(key='timestamp') }}"))

# estimator for LSTM model training
import sagemaker
import os
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

bucket = sagemaker.Session().default_bucket()
role = get_execution_role()

_output_path2 = 's3://{}/{}/training-output-LSTM'.format(bucket, prefix)

_hyperparameters2 = {'activation': 'relu',
                     'cell_num': 12,
                     'loss': 'mean_squared_error',
                     'data_dir': '/opt/ml/input/data',
                     'output_dir': '/opt/ml/model',
                     'checkpoint_dir': '/opt/ml/checkpoints',
                     'optimizer': 'adam',
                     'epochs': 10,
                     'batch_size': 1}

estimator_lstm = TensorFlow(entry_point='model-lstm.py',
                        source_dir='codes',
                        git_config=_git_config,
                        base_job_name='airflow-lstm-training',
                        instance_type='ml.p3.2xlarge',  # V100 单卡
                        image_uri=model_lstm_training_image_uri,
                        instance_count=1,
                        hyperparameters=_hyperparameters2,
                        role=role,
                        metric_definitions=[{'Name': 'train:loss', 'Regex': '.*loss: ([0-9\\.]+)'}],
                        output_path=_output_path2,
                        framework_version='2.1.0',
                        checkpoint_local_path='/opt/ml/checkpoints/',
                        checkpoint_s3_uri=_output_path2 + '/checkpoints',
                        debugger_hook_config=False,
                        py_version='py3')

data_input_dic2 = {'train': output_lstm_train_data, 'test': output_lstm_test_data}

# train_config specifies SageMaker training configuration for training operator
_train_config_lstm = training_config(estimator=estimator_lstm,
                                inputs=data_input_dic2,
                                job_name='airflow-lstm-training-{}'.format(
                                    "{{ task_instance.xcom_pull(key='timestamp') }}"))


# evaluation of 2 models
def evaluation(**context):
    _job_name = 'airflow-evaluation-{}'.format(context['task_instance'].xcom_pull(key='timestamp'))

    output_prefix = 'jiuzhang/evaluation-result'

    input_model1 = 's3://{}/{}/{}/{}/{}/'.format(bucket, prefix, 'training-output-NN',
                                                 'airflow-nn-training-{}'.format(
                                                     context['task_instance'].xcom_pull(key='timestamp')), 'output')
    input_model2 = 's3://{}/{}/{}/{}/{}/'.format(bucket, prefix, 'training-output-LSTM',
                                                 'airflow-lstm-training-{}'.format(
                                                     context['task_instance'].xcom_pull(key='timestamp')), 'output')

    input_nn_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix, 'nn-test')
    input_lstm_test_data = 's3://{}/{}/{}/'.format(bucket, input_preprocessed_prefix, 'lstm-test')

    output_evaluation_result = 's3://{}/{}/'.format(bucket, output_prefix)

    from sagemaker.processing import ProcessingInput, ProcessingOutput

    _inputs = [ProcessingInput(source=input_model1,
                               destination='/opt/ml/processing/input-model1',
                               s3_data_distribution_type='ShardedByS3Key'),
               ProcessingInput(source=input_model2,
                               destination='/opt/ml/processing/input-model2',
                               s3_data_distribution_type='ShardedByS3Key'),
               ProcessingInput(source=input_nn_test_data,
                               destination='/opt/ml/processing/input-dataset1',
                               s3_data_distribution_type='ShardedByS3Key'),
               ProcessingInput(source=input_lstm_test_data,
                               destination='/opt/ml/processing/input-dataset2',
                               s3_data_distribution_type='ShardedByS3Key'),
               ]

    _outputs = [ProcessingOutput(output_name='evaluatioin_result',
                                 source='/opt/ml/processing/output',
                                 destination=output_evaluation_result)]

    sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                         role=role,
                                         instance_type='ml.m5.xlarge',
                                         instance_count=1)

    sklearn_processor.run(code='s3://sagemaker-cn-northwest-1-542319707026/sagemaker-repo/codes/evaluation.py',  # 自带的处理代码
                          inputs=_inputs,
                          outputs=_outputs,
                          job_name=_job_name,
                          arguments=['--sns-topic', 'arn:aws-cn:sns:cn-northwest-1:542319707026:sm-airflow-evaluation']
                          )

    return _job_name

# pending for manual model approval
def model_approval(**context):
    raise AirflowException("Please change this step to success to continue")

# create model config for model operator
# model creation
_model_config_nn = model_config_from_estimator(estimator=estimator_nn,
                                            task_id='nn_training',
                                            task_type='training',
                                            instance_type = 'ml.c5.2xlarge',
                                            name='model-lstm-{}'.format("{{ task_instance.xcom_pull(key='timestamp') }}"),
                                            role=role_arn,
                                            image_uri=model_inference_image_uri)

_model_config_lstm = model_config_from_estimator(estimator=estimator_lstm,
                                            task_id='lstm_training',
                                            task_type='training',
                                            instance_type = 'ml.c5.2xlarge',
                                            name='model-nn-{}'.format("{{ task_instance.xcom_pull(key='timestamp') }}"),
                                            role=role_arn,
                                            image_uri=model_inference_image_uri)

# Model Registry
def model_registry(**context):
    print('this method is for model registry to parameter store')

    ## get status of approval
    _model_flag = None
    task_approval_nn = dag.get_task(task_id='manual_model_approval_nn')
    task_approval_lstm = dag.get_task(task_id='manual_model_approval_lstm')
    state_nn = TaskInstance(task_approval_nn, context["execution_date"]).current_state()
    state_lstm = TaskInstance(task_approval_lstm, context["execution_date"]).current_state()

    _model_flag = 'nn' if state_nn == 'success' else 'lstm'

    _model_name = 'model-{}-{}'.format(_model_flag,context['task_instance'].xcom_pull(key='timestamp'))

    response = client_ssm.put_parameter(
        Name='/model-repo/latest-time-series-model',
        Type='String',
        Value=_model_name,
        Overwrite=True
    )

    print(_model_flag)
    return _model_flag

# -------
# airflow pipeline
# -------

set_timestamp_op = PythonOperator(
    task_id='config_setting',
    python_callable=config_timestamp,
    op_args=[],
    provide_context=True,
    dag=dag)

processing_op = PythonOperator(
    task_id='preprocessing',
    python_callable=preprocessing,
    op_args=[],
    provide_context=True,
    retries = 5,
    retry_delay=timedelta(minutes=10),
    dag=dag)

train_nn_op = SageMakerTrainingOperator(
    task_id='nn_training',
    config=_train_config_nn,
    wait_for_completion=True,
    dag=dag)

train_lstm_op = SageMakerTrainingOperator(
    task_id='lstm_training',
    config=_train_config_lstm,
    wait_for_completion=True,
    dag=dag)

evaluating_op = PythonOperator(
    task_id='evaluating',
    python_callable=evaluation,
    op_args=[],
    provide_context=True,
    retries = 5,
    retry_delay=timedelta(minutes=10),
    dag=dag)

approve_lstm_op = PythonOperator(
    task_id='manual_model_approval_lstm',
    python_callable=model_approval,
    op_args=[],
    provide_context=True,
    retries = 60,
    retry_delay=timedelta(minutes=1),
    dag=dag)

approve_nn_op = PythonOperator(
    task_id='manual_model_approval_nn',
    python_callable=model_approval,
    op_args=[],
    provide_context=True,
    retries = 60,
    retry_delay=timedelta(minutes=1),
    dag=dag)

model_registry_op = PythonOperator(
    task_id='model_registry',
    python_callable=model_registry,
    op_args=[],
    trigger_rule = 'one_success',
    provide_context=True,
    dag=dag)

endpoint_creation_op = SageMakerEndpointOperator(
    task_id='model_deployment',
    config= deploy_config(
        model=Model(image_uri=model_inference_image_uri,
                  model_data='s3://{}/{}/{}/{}/{}/model.tar.gz'.format(bucket, prefix, 'training-output-NN',
                                                                       'airflow-{}-training-{}'.format("{{ task_instance.xcom_pull(task_ids='model_registry') }}","{{ task_instance.xcom_pull(key='timestamp') }}"),
                                                                    'output'),
                  role=role_arn,
                  name='model-{}-{}'.format("{{ task_instance.xcom_pull(task_ids='model_registry') }}","{{ task_instance.xcom_pull(key='timestamp') }}")),
        initial_instance_count=1,
        instance_type='ml.c5.xlarge',
        endpoint_name='endpoint-{}'.format("{{ task_instance.xcom_pull(key='timestamp') }}")),
    wait_for_completion=True,
    check_interval=30,
    max_ingestion_time=None,
    provide_context=True,
    dag=dag)

processing_op.set_upstream(set_timestamp_op)
train_nn_op.set_upstream(processing_op)
train_lstm_op.set_upstream(processing_op)
evaluating_op.set_upstream([train_nn_op, train_lstm_op])
approve_lstm_op.set_upstream(evaluating_op)
approve_nn_op.set_upstream(evaluating_op)
model_registry_op.set_upstream([approve_lstm_op, approve_nn_op])
endpoint_creation_op.set_upstream(model_registry_op)

print('tes11')