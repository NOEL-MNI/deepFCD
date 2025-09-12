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

# specify miniforge version
ARG MINIFORGE_VERSION=25.3.1-0
RUN wget https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh \
    && bash Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh -b -p "${HOME}/conda" \
    && rm Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh

# copy requirements early for better layer caching
COPY app/requirements.txt /app/requirements.txt

# create conda environment
RUN eval "$(conda shell.bash hook)" \
    && conda create -n deepFCD -c conda-forge python=3.8.20 pygpu==0.7.6 pyyaml\<6.0 \
    && conda clean -a -y

# install pip packages in separate layer
RUN eval "$(conda shell.bash hook)" \
    && conda activate deepFCD \
    && python -m pip install --no-cache-dir -r /app/requirements.txt \
    && conda deactivate

# clean up caches
RUN eval "$(conda shell.bash hook)" \
    && conda activate deepFCD \
    && conda clean -a -y \
    && python -m pip cache purge

COPY app/ /app/

COPY tests/ /tests/

RUN sudo chmod -R 777 /app && sudo chmod +x /app/inference.py

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "deepFCD"]

CMD ["python3"]