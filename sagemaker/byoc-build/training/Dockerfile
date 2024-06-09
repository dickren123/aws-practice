FROM 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/tensorflow-training:2.1.0-gpu-py36-cu101-ubuntu18.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         cmake \
         libopenblas-dev \
         liblapack-dev \
         libx11-dev \
         libgtk-3-dev \
         python3-setuptools \
         python python-dev python-pip \
         python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN echo "It is a test1"

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple sagemaker-containers setuptools numpy dlib