# based on the cuda image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# install python 3.10
RUN apt-get update \
    && apt-get install -y git curl vim wget python3.10 libpython3.10-dev python3-pip \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python3.10  /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10  /usr/bin/python \
    && apt-get autoclean && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip
# install 3rd-party system dependencies, cmake
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 software-properties-common build-essential cmake  -y

# install java for quality classifier
# prepare the java env
WORKDIR /opt
# download jdk
RUN wget https://aka.ms/download-jdk/microsoft-jdk-17.0.9-linux-x64.tar.gz -O jdk.tar.gz && \
    tar -xzf jdk.tar.gz && \
    rm -rf jdk.tar.gz && \
    mv jdk-17.0.9+8 jdk
# set the environment variable
ENV JAVA_HOME=/opt/jdk

# install for better synth
WORKDIR /dj_synth_challenge
COPY . .
RUN bash install.sh
