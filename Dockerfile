FROM noelmni/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>" \
        org.opencontainers.image.title="deepFCD" \
        org.opencontainers.image.description="Automated Detection of Focal Cortical Dysplasia using Deep Learning" \
        org.opencontainers.image.licenses="BSD-3-Clause" \
        org.opencontainers.image.source="https://github.com/NOEL-MNI/deepFCD" \
        org.opencontainers.image.url="https://github.com/NOEL-MNI/deepFCD"

# manually update outdated repository key
# fixes invalid GPG error: https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git \
    bash \
    wget \
    bzip2 \
    sudo \
    && sudo apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH=/home/user/conda/bin:${PATH}

# create a working directory
RUN mkdir /app
WORKDIR /app

# create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# all users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh \
    && /bin/bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b -p /home/user/conda \
    && rm Miniconda3-py38_23.5.2-0-Linux-x86_64.sh

# RUN conda update -n base -c defaults conda

RUN git clone --depth 1 https://github.com/NOEL-MNI/deepMask.git \
    && rm -rf deepMask/.git

RUN eval "$(conda shell.bash hook)" \
    && conda create -n preprocess python=3.8 \
    && conda activate preprocess \
    && python -m pip install -r deepMask/app/requirements.txt \
    && conda deactivate

COPY app/requirements.txt /app/requirements.txt

RUN python -m pip install -r /app/requirements.txt \
    && conda install -c conda-forge pygpu==0.7.6 \
    && pip cache purge

COPY app/ /app/

RUN sudo chmod -R 777 /app && sudo chmod +x /app/inference.py

CMD ["python3"]