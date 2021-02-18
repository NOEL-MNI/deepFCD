<h1 align="center">
  <b>Multicenter Validated Detection of Focal Cortical Dysplasia using Deep Learning</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.6-ff69b4.svg" /></a>
      <a href= "https://keras.io/">
        <img src="https://img.shields.io/badge/Keras-2.2.4-2BAF2B.svg" /></a>
      <a href= "https://github.com/Theano/Theano">
        <img src="https://img.shields.io/badge/Theano-1.0.4-2BAF2B.svg" /></a>
      <a href= "https://github.com/NOEL-MNI/deepFCD/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
      <a href= "https://zenodo.org/deposit/4521706">
        <img src="https://zenodo.org/badge/4521706.svg" /></a>
</p>


------------------------

![](assets/diagram.jpg)

### Please cite:
```TeX
@misc{Gill2021,
  author = {Gill RS, et al},
  title = {Multicenter Validated Detection of Focal Cortical Dysplasia using Deep Learning},
  year = {2021},
  publisher = {Americal Academy of Neurology},
  journal = {Neurology},
  howpublished = {\url{https://github.com/NOEL-MNI/deepFCD}},
  doi = {10.5281/zenodo.4521706}
}
```

## Pre-requisites
###TODO: Update version requirements
```Shell
1. Python == 3.6
2. Keras == 2.2.4
3. Theano == 1.0.4
4. Miniconda3
```

## Installation

```Shell
conda create -n deepFCD python=3.8
conda activate deepFCD
pip install -r app/requirements.txt
```


## Usage
###TODO: Training and Inference
### Docker
```Shell
docker run -it -v /tmp:/tmp docker.pkg.github.com/noel-mni/deepfcd/app:latest /app/inference.py \
                                            $PATIENT_ID \
                                            /tmp/T1.nii.gz /tmp/FLAIR.nii.gz \
                                            /tmp
```

## License
<a href= "https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
```Shell
Copyright 2021 Neuroimaging of Epilepsy Laboratory, McGill University
```
