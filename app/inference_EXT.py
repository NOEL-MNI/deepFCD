#!/usr/bin/env python3

import os, sys, socket, csv, time, fnmatch, re, string, json

from config.experiment import options
hostname = socket.getfqdn()
options['hostname'] = hostname

print('-'*60)
print("hostname : {}".format(hostname))
print('-'*60)

os.environ["KERAS_BACKEND"] = "theano"

if hostname.startswith("pandarus") or hostname.startswith("hamlet") or hostname.startswith("silius"):
    options['cuda'] = sys.argv[1] # flag using gpu 1 or 2
    if options['cuda'].startswith('cuda1'):
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1,floatX=float32"
    elif options['cuda'].startswith('cpu'):
        os.environ["OMP_NUM_THREADS"]="32"
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,openmp=True,floatX=float32"
    else:
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"

print(os.environ["THEANO_FLAGS"])

import numpy as np
from nibabel import load as load_nii
import pandas as pd
import setproctitle as spt
from tqdm import tqdm

from models.noel_models_keras import *
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K

from utils.metrics import *
from utils.base import *

K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')  # TH dimension ordering in this code

test_site = 'TLE'

options['parallel_gpu'] = False
modalities = ['T1', 'FLAIR']
x_names = options['x_names']
options['y_names'] = ['_lesion.nii.gz']
y_names = options['y_names']

# seed = options['seed']

options['dropout_mc'] = True
options['batch_size'] = 350000
options['mini_batch_size'] = 2048

options['load_checkpoint_1'] = True
options['load_checkpoint_2'] = True
# options['continue_training_2'] = True
# options['initial_epoch_2'] = 69

x_names = options['x_names']
y_names = ['_lesion.nii.gz', 'lesion_bin.nii.gz']
# Select an experiment name to store net weights and segmentation masks
options['experiment'] = 'exp_dropoutMC_' + test_site

print("experiment: {}".format(options['experiment']))
spt.setproctitle(options['experiment'])

options['model_dir'] = '/host/hamlet/local_raid/data/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/models'

# --------------------------------------------------
# initialize the CNN
# --------------------------------------------------
options['weight_paths'] = os.path.join(options['model_dir'], options['experiment'])

model = None  # Clearing the CNN.

model = off_the_shelf_model(options)

print(model[0].summary())

load_weights = os.path.join(options['weight_paths'], options['experiment']+'_model_1.h5')
print("loading DNN1, model[0]: {} exists".format(load_weights)) if os.path.isfile(load_weights) else sys.exit("model[0]: {} doesn't exist".format(load_weights))
model[0] = load_model(load_weights)

load_weights = os.path.join(options['weight_paths'], options['experiment']+'_model_2.h5')
print("loading DNN1, model[1]: {} exists".format(load_weights)) if os.path.isfile(load_weights) else sys.exit("model[1]: {} doesn't exist".format(load_weights))
model[1] = load_model(load_weights)
print('model #2 loaded')

print(model[0].summary())


# --------------------------------------------------
# test the cascaded model
# --------------------------------------------------

print('-----------------------------------')
print("testing new patient/s")
print('-----------------------------------')

test_list = []
test_data = {}
# test controls
tfolder = '/host/hamlet/local_raid/data/ravnoor/sandbox/deepMasks'

# test_list = re.findall(r'[$0]\d{3}_\d{1}', ''.join(test_list)) # match 0303_1, etc.
include = ['ESP_001']

list_of_scans = os.listdir(tfolder)
test_list = [e for e in list_of_scans if e in include]
test_list.sort()
test_data = {f: {m: os.path.join(tfolder, f, m+'_stripped.nii.gz') for m in modalities} for f in test_list}

# for scan in test_list:
for _, scan in enumerate(tqdm(test_list, desc='serving predictions using the trained model')):
# for _, scan in enumerate(test_list):

    t_data = {}
    t_data[scan] = test_data[scan]

    # test_folder = '/host/hamlet/local_raid/data/ravnoor/sandbox/'
    test_folder = tfolder
    options['pred_folder'] = os.path.join(test_folder, scan, options['experiment'])

    pred_mean_fname = os.path.join(options['pred_folder'], options['experiment'] + '_prob_mean_1.nii.gz')
    pred_var_fname = os.path.join(options['pred_folder'], options['experiment'] + '_prob_var_1.nii.gz')

    if np.logical_and(os.path.isfile(pred_mean_fname), os.path.isfile(pred_var_fname)):
        print("prediction for {} already exists".format(scan))
        continue

    if not os.path.exists(options['pred_folder']):
        os.path.join(test_folder, options['experiment'])
        os.mkdir(options['pred_folder'])
        # os.mkdir(os.path.join(test_folder, options['experiment']))
        # os.mkdir(os.path.join(test_folder, options['experiment'], scan))

    options['test_name'] = scan + '_' + options['experiment'] + '.nii.gz'
    options['test_scan'] = scan

    start = time.time()

    print('\n')
    print('-'*80)
    print("testing the model for scan: {}".format(scan))
    print('-'*80)

    test0, test1, test2, _, _ = test_model(model, t_data, options)   # test0: prediction/stage1 | test1: pred/stage2 | test2: morphological processing + clustered | lpred: predicted label only | count: # false positives
