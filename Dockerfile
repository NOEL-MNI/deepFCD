FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>" \
        org.opencontainers.image.title="deepFCD" \
        org.opencontainers.image.description="Automated Detection of Focal Cortical Dysplasia using Deep Learning" \
        org.opencontainers.image.licenses="BSD-3-Clause" \
        org.opencontainers.image.source="https://github.com/NOEL-MNI/deepFCD" \
        org.opencontainers.image.url="https://github.com/NOEL-MNI/deepFCD"

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

# specify conda version
ARG CONDA_VERSION=py38_23.11.0-2
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh \
    && /bin/bash Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -b -p /home/user/conda \
    && rm Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh

# RUN conda update -n base -c defaults conda

RUN git clone --depth 1 https://github.com/NOEL-MNI/deepMask.git \
    && rm -rf deepMask/.git

RUN eval "$(conda shell.bash hook)" \
    && conda create -n preprocess python=3.8 \
    && conda activate preprocess \
    && python -m pip install -r deepMask/app/requirements.txt \
    && conda deactivate

RUN eval "$(conda shell.bash hook)" \
    && conda activate base \
    && python -m pip install conda-lock

COPY app/conda-lock.yml /app/conda-lock.yml

RUN eval "$(conda shell.bash hook)" \
    && conda run --name base conda-lock install --name deepFCD /app/conda-lock.yml

COPY app/ /app/

COPY tests/ /tests/

RUN sudo chmod -R 777 /app && sudo chmod +x /app/inference.py

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "deepFCD"]

CMD ["python3"]