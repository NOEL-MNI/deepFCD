<h1 align="center">
  <b>Multicenter Validated Detection of Focal Cortical Dysplasia using Deep Learning</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.7-ff69b4.svg" /></a>
      <a href= "https://keras.io/">
        <img src="https://img.shields.io/badge/Keras-2.2.4-2BAF2B.svg" /></a>
      <a href= "https://github.com/Theano/Theano">
        <img src="https://img.shields.io/badge/Theano-1.0.4-2BAF2B.svg" /></a>
      <a href= "https://github.com/NOEL-MNI/deepFCD/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
      <a href="https://doi.org/10.5281/zenodo.4521706">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4521706.svg" alt="DOI"></a>
</p>


------------------------

![](assets/diagram.jpg)

### Please cite:
```TeX
@article{GillFCD2021,
  title = {Multicenter Validated Detection of Focal Cortical Dysplasia using Deep Learning},
  author = {Gill, Ravnoor Singh and Lee, Hyo-Min and Caldairou, Benoit and Hong, Seok-Jun and Barba, Carmen and Deleo, Francesco and D'Incerti, Ludovico and Coelho, Vanessa Cristina Mendes and Lenge, Matteo and Semmelroch, Mira and others},
  journal = {Neurology},
  year = {2021},
  publisher = {Americal Academy of Neurology},
  code = {\url{https://github.com/NOEL-MNI/deepFCD}},
  doi = {https://doi.org/10.1212/WNL.0000000000012698}
}
```

## Pre-requisites
```bash
0. Anaconda Python Environment
1. Python == 3.7.x
2. Keras == 2.2.4
3. Theano == 1.0.4
4. ANTsPy == 0.3.2 (for MRI preprocessing)
5. PyTorch == 1.4.0 (for deepMask)
6. h5py == 2.10.0
+ app/requirements.txt
```

## Installation

```bash
# install Miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

# create and activate a Conda environment
conda create -n deepFCD python=3.7
conda activate deepFCD

# install dependencies using pip
python -m pip install -r app/requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge pygpu=0.7.6
```


## Usage
### TODO: Training routine

### Inference
```bash
chmod +x ./app/inference.py # make the script executable -ensure you have the requisite permissions
./app/inference.py     \ # the script to perform inference on the multimodal MRI images
    ${PATIENT_ID}      \ # prefix for the filenames; for example: FCD_001 (needed for outputs only)
    ${T1_IMAGE}        \ # T1-weighted image; for example: FCD_001_t1.nii.gz or t1.nii.gz [T1 is specified before FLAIR - order is important]
    ${FLAIR_IMAGE}     \ # T2-weighted FLAIR image; for example: FCD_001_t2.nii.gz or flair.nii.gz [T1 is specified before FLAIR - order is important]
    ${INPUT_DIRECTORY}   # input/output directory
```
### Inference using Docker
```bash
docker run --rm -it --init \
    --gpus=all                 \ # expose the host GPUs to the guest docker container
    --user="$(id -u):$(id -g)" \ # map user permissions appropriately
    --volume="$PWD:/io"        \ # $PWD refers to the present working directory containing the input images, can be modified to a local host directory
    noelmni/deep-fcd:latest    \ # docker image containing all the necessary software dependencies
    /app/inference.py  \ # the script to perform inference on the multimodal MRI images
    ${PATIENT_ID}      \ # prefix for the filenames; for example: FCD_001 (needed for outputs only)
    ${T1_IMAGE}        \ # T1-weighted image; for example: FCD_001_t1.nii.gz or t1.nii.gz [T1 is specified before FLAIR - order is important]
    ${FLAIR_IMAGE}     \ # T2-weighted FLAIR image; for example: FCD_001_t2.nii.gz or flair.nii.gz [T1 is specified before FLAIR - order is important]
    /io                  # input/output directory within the container mapped to ${PWD} [ DO NOT MODIFY]
```

## License
<a href= "https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>

```console
Copyright 2021 Neuroimaging of Epilepsy Laboratory, McGill University
```
