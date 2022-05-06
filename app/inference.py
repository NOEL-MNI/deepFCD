#!/usr/bin/env python3

import os
import sys
import multiprocessing
from mo_dots import Data
import subprocess
from config.experiment import options
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import setproctitle as spt
from tqdm import tqdm
from utils.helpers import *

os.environ["KERAS_BACKEND"] = "theano"

# GPU/CPU options
options['cuda'] = sys.argv[5] # cpu, cuda, cuda0, cuda1, or cudaX: flag using gpu 1 or 2
if options['cuda'].startswith('cuda1'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1,floatX=float32,dnn.enabled=False"
elif options['cuda'].startswith('cpu'):
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count() // 2)
    var = os.getenv('OMP_NUM_THREADS', None)
    try:
        print("# of threads initialized: {}".format(int(var)))
    except ValueError:
        raise TypeError("The environment variable OMP_NUM_THREADS"
                        " should be a number, got '%s'." % var)
    # os.environ['openmp'] = 'True'
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,openmp=True,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32,dnn.enabled=False"
print(os.environ["THEANO_FLAGS"])

from models.noel_models_keras import *
from keras.models import load_model
from keras import backend as K
from utils.metrics import *
from utils.base import *


# configuration
args = Data()
args.id = sys.argv[1]
args.t1_fname = sys.argv[2]
args.t2_fname = sys.argv[3]
args.dir = sys.argv[4]
args.brain_masking = True # set to True or any non-zero value for brain extraction or skull-removal, False otherwise
args.preprocess = False # co-register T1 and T2 contrasts before brain extraction
args.outdir = os.path.join(args.dir, args.id)

args.t1 = os.path.join(args.outdir, args.t1_fname)
args.t2 = os.path.join(args.outdir, args.t2_fname)
cwd = os.path.dirname(__file__)

if args.brain_masking:
    args.use_gpu = False
    # MRI pre-processing configuration
    args.output_suffix = '_brain_final.nii.gz'

    preprocess_sh = os.path.join(cwd, 'preprocess.sh')
    subprocess.check_call([preprocess_sh, args.id, args.t1_fname, args.t2_fname, args.dir, bool2str(args.preprocess), bool2str(args.use_gpu)])

    args.t1 = os.path.join(args.outdir, args.id + '_t1' + args.output_suffix)
    args.t2 = os.path.join(args.outdir, args.id + '_t2' + args.output_suffix)
else:
    print("Skipping image preprocessing and brain masking, presumably images are co-registered, bias-corrected, and skull-stripped")

# deepFCD configuration
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

# trained model weights based on 148 histologically-verified FCD subjects
options['test_folder'] = args.dir
options['weight_paths'] = os.path.join(cwd, 'weights')
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
# test_list = ['mcd_0468_1']
test_list = [args.id]
files = [args.t1, args.t2]
test_data = {}
test_data = {f: {m: os.path.join(options['test_folder'], f, n) for m, n in zip(modalities, files)} for f in test_list}

for _, scan in enumerate(tqdm(test_list, desc='serving predictions using the trained model', colour='blue')):
    t_data = {}
    t_data[scan] = test_data[scan]

    options['pred_folder'] = os.path.join(options['test_folder'], scan, options['experiment'])
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