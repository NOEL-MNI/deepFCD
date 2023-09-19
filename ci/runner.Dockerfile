FROM noelmni/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>" \
        org.opencontainers.image.title="Self-hosted Github Actions runner for deepFCD" \
        org.opencontainers.image.description="Automated Detection of Focal Cortical Dysplasia using Deep Learning" \
        org.opencontainers.image.licenses="BSD-3-Clause" \
        org.opencontainers.image.source="https://github.com/NOEL-MNI/deepFCD" \
        org.opencontainers.image.url="https://github.com/NOEL-MNI/deepFCD"

# manually update outdated repository key
# fixes invalid GPG error: https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

ARG RUNNER_VERSION=2.309.0
ARG NVM_VERSION=0.39.5

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
    bash build-essential curl jq libssl-dev libffi-dev \
    nano python3-dev software-properties-common unzip wget

# install git 2.17+
RUN add-apt-repository ppa:git-core/candidate -y
RUN apt-get update
RUN apt-get install -y git

RUN apt-get remove nodejs npm

# github actions needs a non-root to run
RUN useradd -m ga 
WORKDIR /home/ga/actions-runner
ENV HOME=/home/ga

# https://stackoverflow.com/questions/25899912/how-to-install-nvm-in-docker/60137919#60137919
SHELL ["/bin/bash", "--login", "-i", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v${NVM_VERSION}/install.sh | bash
RUN source /root/.bashrc && nvm install 16
SHELL ["/bin/bash", "--login", "-c"]

# install Github Actions runner
RUN curl -s -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    ./bin/installdependencies.sh

# RUN wget -c https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz && \
#     ./bin/patchelf --set-interpreter /opt/glibc-2.28/lib/ld-linux-x86-64.so.2 --set-rpath /opt/glibc-2.28/lib/ /home/ga/.nvm/versions/node/v20.6.1/bin/node

# add over the start.sh script
ADD start-runner.sh start.sh

# make the script executable
RUN chmod +x start.sh

# set permission and user to ga
RUN chown -R ga /home/ga
USER ga

# install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh && \
    /bin/bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b && \
    rm Miniconda3-py38_23.5.2-0-Linux-x86_64.sh && \
    echo '. ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc

# create a dir to store inputs and outputs
RUN mkdir ~/io

# set the entrypoint to the start.sh script
ENTRYPOINT ["./start.sh"]