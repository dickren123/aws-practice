import sagemaker

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = "sagemaker/DEMO-pytorch-mnist"

role = 'arn:aws-cn:iam::542319707026:role/service-role/AmazonSageMaker-ExecutionRole-20200403T105376'

from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="start.sh",
    source_dir="../codes",
    role=role,
    framework_version="1.8",
    instance_count=1,
    instance_type="ml.p3.8xlarge",
    py_version='py3',
    hyperparameters={"epochs": 100, "backend": "nccl"},
)

estimator.fit({"training": 's3://sagemaker-cn-northwest-1-542319707026/sagemaker/DEMO-pytorch-mnist'})

