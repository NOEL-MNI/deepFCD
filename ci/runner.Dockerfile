FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>" \
        org.opencontainers.image.title="Self-hosted Github Actions runner for deepFCD" \
        org.opencontainers.image.description="Automated Detection of Focal Cortical Dysplasia using Deep Learning" \
        org.opencontainers.image.licenses="BSD-3-Clause" \
        org.opencontainers.image.source="https://github.com/NOEL-MNI/deepFCD" \
        org.opencontainers.image.url="https://github.com/NOEL-MNI/deepFCD"

ARG RUNNER_VERSION=2.328.0

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
RUN curl -s -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    ./bin/installdependencies.sh && \
    rm -f ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# add the start.sh script
ADD start-runner.sh start.sh

# make the script executable
RUN chmod +x start.sh

# set permission and user to ga
RUN chown -R ga /home/ga
USER ga

# specify miniforge version
ARG MINIFORGE_VERSION=25.3.1-0
RUN curl -s -O -L https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh \
    && bash Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh -b -p "${HOME}/conda" \
    && rm Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh
    
# initialize conda
RUN echo 'source "${HOME}/conda/etc/profile.d/conda.sh"' >> ~/.bashrc

# create a dir to store inputs and outputs
RUN mkdir ${HOME}/io

# set the entrypoint to the start.sh script
ENTRYPOINT ["./start.sh"]