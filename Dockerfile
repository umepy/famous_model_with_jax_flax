FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install python3.9
RUN apt-get -y install python3-pip
RUN apt-get -y install wget curl tzdata git libgl1-mesa-dev
RUN apt-get -y install libopencv-dev

WORKDIR /work

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install opencv-python numpy tqdm tensorboard torchinfo black flake8 isort

# install jax
RUN python3.9 -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN python3.9 -m pip install flax

# install pytorch
RUN python3.9 -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html