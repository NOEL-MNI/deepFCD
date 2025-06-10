ARG BASE_IMAGE_TAG=11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${BASE_IMAGE_TAG}

LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>" \
        org.opencontainers.image.title="Self-hosted Github Actions runner for deepFCD" \
        org.opencontainers.image.description="Automated Detection of Focal Cortical Dysplasia using Deep Learning" \
        org.opencontainers.image.licenses="BSD-3-Clause" \
        org.opencontainers.image.source="https://github.com/NOEL-MNI/deepFCD" \
        org.opencontainers.image.url="https://github.com/NOEL-MNI/deepFCD"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    git \
    jq \
    libffi-dev \
    libssl-dev \
    nano \
    unzip

# github actions needs a non-root to run
RUN useradd -m ga 
WORKDIR /home/ga/actions-runner
ENV HOME=/home/ga

# install Github Actions runner
ARG RUNNER_VERSION=2.321.0
RUN curl -s -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    ./bin/installdependencies.sh

# add over the start.sh script
ADD start-runner.sh start.sh

# make the script executable
RUN chmod +x start.sh

# install micromamba
ARG MAMBA_VERSION=2.0.5-0
RUN curl -L -o /usr/bin/micromamba https://github.com/mamba-org/micromamba-releases/releases/download/${MAMBA_VERSION}/micromamba-linux-64 && \
    chmod +x /usr/bin/micromamba

# set permission and user to ga
RUN chown -R ga /home/ga
USER ga

# initialize micromamba
ENV MAMBA_EXE=/usr/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/home/ga/micromamba
ENV CONDA_DEFAULT_ENV=base
ENV MAMBA_USER=ga
RUN micromamba shell init --shell bash
# this activates the base environment with py38
RUN eval "$(micromamba shell hook -s posix)" && \
    micromamba activate && \
    micromamba install python=3.8 -c conda-forge

# activate micromamba base env
RUN echo "micromamba activate" >> ~/.bashrc

# set aliases for conda
RUN echo "alias mamba='/usr/bin/micromamba'" >> ~/.bash_aliases && \
    echo "alias conda='/usr/bin/micromamba'" >> ~/.bash_aliases

# create a dir to store inputs and outputs
RUN mkdir ~/io

# set the entrypoint to the start.sh script
ENTRYPOINT ["./start.sh"]