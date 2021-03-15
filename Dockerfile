FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>"

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git \
                        wget \
                        bzip2 \
                        build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH=/opt/conda/bin:${PATH}

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
 && /bin/bash Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -p /opt/conda \
 && rm Miniconda3-py37_4.9.2-Linux-x86_64.sh

RUN conda install --yes Theano=1.0.4 keras=2.2.4 -c conda-forge && conda install pytorch torchvision torchaudio cpuonly -c pytorch

COPY app/ /app/

RUN pip install -r /app/requirements.txt