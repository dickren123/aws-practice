#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sagemaker import get_execution_role
import os
import sagemaker
from sagemaker.tensorflow import TensorFlow

_role = 'arn:aws:iam::907488872981:role/service-role/AmazonSageMaker-ExecutionRole-20190926T171845'
print(sagemaker.__version__)

# In[2]:


# upgrade sagemaker sdk version
# !pip install sagemaker --upgrade

# Setup local training env in this notebook
# !wget -q https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-script-mode/master/local_mode_setup.sh
# !wget -q https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-script-mode/master/daemon.json    


# In[3]:


# !/bin/bash ./local_mode_setup.sh


# In[5]:


_hyperparameters = {'task_type':'train',
                   'learning_rate':0.0005,
                   'optimizer':'Adam',
                   'num_epochs':1,
                   'batch_size':256,
                   'field_size':39,
                   'feature_size':117581,
                   'deep_layers':'400,400,400',
                   'dropout':'0.5,0.5,0.5',
                   'ckpt_dir':'/opt/ml/model',
                   'learning_rate':0.0001,
                   'checkpoint_steps':100}


# In[7]:


# _entry_file = '0-deepfm-local.py'
# _tf_version = '1.15.2'
_entry_file = '0-deepfm-tf1-1i-default.py'
_tf_version = '1.12'
_instance_type = 'local'

_dataset_dir = os.path.join(os.getcwd(), 'data-prepare/raw-cutoff-100000')
_output_dir = os.path.join(os.getcwd(), 'local-output')


# In[8]:


_metric_definition = [
    {'Name': 'train:loss', 'Regex': '.*loss = ([0-9\\.]+)'},
    {'Name': 'train:global_step', 'Regex': '.*global_step = ([0-9\\.]+),.*'},
    {'Name': 'validation:accuracy', 'Regex': '.*auc = ([0-9\\.]+), global_step.*'}
]


# In[9]:


estimator = TensorFlow(entry_point=_entry_file,
                       source_dir='code',
                       instance_type=_instance_type,
                       instance_count=1,
                       hyperparameters=_hyperparameters,
                       metric_definitions=_metric_definition,
                       role=_role,
                       output_path = f'file://{_output_dir}',
                       framework_version=_tf_version,
                       py_version='py3')

_inputs = {'datasets': f'file://{_dataset_dir}'}
estimator.fit(_inputs)

