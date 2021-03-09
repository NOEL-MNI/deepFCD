#!/usr/bin/env python3

import os
import sys
import multiprocessing
from config.experiment import options
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import setproctitle as spt
from tqdm import tqdm

os.environ["KERAS_BACKEND"] = "theano"

options['cuda'] = sys.argv[1] # flag using gpu 1 or 2
if options['cuda'].startswith('cuda1'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1,floatX=float32"
elif options['cuda'].startswith('cpu'):
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['GOTO_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['openmp'] = 'True'
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,openmp=True,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
print(os.environ["THEANO_FLAGS"])

from models.noel_models_keras import *
from keras.models import load_model
from keras import backend as K
from utils.metrics import *
from utils.base import *

K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')  # TH dimension ordering in this code

options['parallel_gpu'] = False
modalities = ['T1', 'FLAIR']
x_names = options['x_names']

# seed = options['seed']
options['dropout_mc'] = True
options['batch_size'] = 350000
options['mini_batch_size'] = 2048
options['load_checkpoint_1'] = True
options['load_checkpoint_2'] = True
options['model_dir'] = '/tmp/models'

# trained model weights based on 148 histologically-verified FCD subjects
options['weight_paths'] = './weights'
options['experiment'] = 'noel_deepFCD_dropoutMC'
print("experiment: {}".format(options['experiment']))
spt.setproctitle(options['experiment'])

# --------------------------------------------------
# initialize the CNN
# --------------------------------------------------
# initialize empty model
model = None
# initialize the CNN architecture
model = off_the_shelf_model(options)

load_weights = os.path.join(options['weight_paths'], 'noel_deepFCD_dropoutMC_model_1.h5')
print("loading DNN1, model[0]: {} exists".format(load_weights)) if os.path.isfile(load_weights) else sys.exit("model[0]: {} doesn't exist".format(load_weights))
model[0] = load_model(load_weights)

load_weights = os.path.join(options['weight_paths'], 'noel_deepFCD_dropoutMC_model_2.h5')
print("loading DNN2, model[1]: {} exists".format(load_weights)) if os.path.isfile(load_weights) else sys.exit("model[1]: {} doesn't exist".format(load_weights))
model[1] = load_model(load_weights)
print(model[1].summary())

# --------------------------------------------------
# test the cascaded model
# --------------------------------------------------
print('-'*70)
print("testing new patient/s")
print('-'*70)

# test controls
options['test_folder'] = '/host/hamlet/local_raid/data/ravnoor/sandbox'
tfolder = '/host/hamlet/local_raid/data/ravnoor/sandbox'

test_list = ['mcd_0468_1']
test_data = {}
test_data = {f: {m: os.path.join(tfolder, f, m+'_stripped.nii.gz') for m in modalities} for f in test_list}

for _, scan in enumerate(tqdm(test_list, desc='serving predictions using the trained model', colour='blue')):
    t_data = {}
    t_data[scan] = test_data[scan]

    test_folder = tfolder
    options['pred_folder'] = os.path.join(test_folder, scan, options['experiment'])
    if not os.path.exists(options['pred_folder']):
        os.mkdir(options['pred_folder'])

    pred_mean_fname = os.path.join(options['pred_folder'], scan + '_prob_mean_1.nii.gz')
    pred_var_fname = os.path.join(options['pred_folder'], scan + '_prob_var_1.nii.gz')

    if np.logical_and(os.path.isfile(pred_mean_fname), os.path.isfile(pred_var_fname)):
        print("prediction for {} already exists".format(scan))
        continue

    options['test_scan'] = scan

    start = time.time()
    print('\n')
    print('-'*70)
    print("testing the model for scan: {}".format(scan))
    print('-'*70)

    # test0: prediction/stage1
    # test1: pred/stage2
    # test2: morphological processing + contiguous clusters
    # pred0, pred1, postproc, _, _ = test_model(model, t_data, options)
    test_model(model, t_data, options)

    end = time.time()
    diff = (end - start) // 60
    print("-"*70)
    print("time elapsed: ~ {} minutes".format(diff))
    print("-"*70)
