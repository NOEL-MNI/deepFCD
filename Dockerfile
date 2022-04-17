FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>"

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

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh \
    && /bin/bash Miniconda3-py37_4.11.0-Linux-x86_64.sh -b -p /home/user/conda \
    && rm Miniconda3-py37_4.11.0-Linux-x86_64.sh

RUN python -m pip install --upgrade --force --ignore-installed pip

COPY app/requirements.txt /app/requirements.txt 

RUN python -m pip install -r /app/requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html

RUN conda install -c conda-forge pygpu==0.7.6

RUN pip cache purge

COPY app/ /app/

RUN sudo chmod -R 777 /app && sudo chmod +x /app/inference.py

CMD ["python3"]