#!/usr/bin/env python

import os, sys, socket, csv, time

hostname = socket.getfqdn()
gpu = 1

print('-'*60)
print("hostname : {}".format(hostname))
print('-'*60)

os.environ["KERAS_BACKEND"] = "theano"
if gpu == '1':
    if hostname.startswith("pandarus"):
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1,floatX=float32"
    else:
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

import numpy as np
import fnmatch
from nibabel import load as load_nii
import pandas as pd
from keras.utils.io_utils import HDF5Matrix

# from models.noel_models_keras import *
# from keras.utils import plot_model
# from keras.models import load_model
from keras import backend as K

from utils.metrics import *
from utils.base import *
from config.experiment import options
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')  # TH dimension ordering in this code

def create_dataset(data_path, X, y):
    import h5py
    f = h5py.File(data_path, 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('data', X.shape, dtype='f')
    X_dset[:] = X
    # Creating dataset to store labels
    y_dset = f.create_dataset('labels', y.shape, dtype='i')
    y_dset[:] = y
    f.close()


holdout_dataset = sys.argv[1]

pattern = holdout_dataset+'*'
modalities = options['modalities']
x_names = options['x_names']
y_names = options['y_names']

seed = options['seed']
print("seed: {}".format(seed))
# Select an experiment name to store net weights and segmentation masks
options['experiment'] = 'exp_LoSo_holdout_'+holdout_dataset.strip('*')
print("experiment: {}".format(options['experiment']))

options['model_dir'] = '/host/silius/local_raid/ravnoor/01_Projects/06_DeepLesion_LoSo/models'

sensitivity = 0
perf1 = {}

list_of_train_scans = os.listdir(options['train_folder']+'brain')
list_of_test_scans = os.listdir(options['train_folder']+'brain')

include_test = fnmatch.filter(list_of_test_scans, pattern)
include_train = list(set(list_of_test_scans).difference(include_test))

modality = [x.lower() for x in modalities]

for m in modality:
    include_train = [string.replace(f, '_'+m+'.nii.gz', '') for f in include_train]
    include_test = [string.replace(f, '_'+m+'.nii.gz', '') for f in include_test]
include_train, include_test = list(set(include_train)), list(set(include_test))

print("training dataset size: {}, testing dataset size: {}".format(len(include_train), len(include_test)))

modality = [x.upper() for x in modality]

for scan in include_train:
    # load paths to all the data
    train_x_data = {f: {m: os.path.join(options['train_folder'], 'brain', f+n) for m, n in zip(modality, x_names)} for f in include_train}
    train_y_data = {f: os.path.join(options['train_folder'], 'lesion_labels/gray_matter', f+y_names[0]) for f in include_train}

# for scan in include_test:
#     # load paths to all the data
#     test_x_data = {f: {m: os.path.join(options['train_folder'], 'brain', f+n) for m, n in zip(modality, x_names)} for f in include_test}
#     test_y_data = {f: os.path.join(options['train_folder'], 'lesion_labels/gray_matter', f+y_names[0]) for f in include_test}

options['weight_paths'] = os.path.join(options['model_dir'], options['experiment'])

print("\n converting 3D MRI to patch-based dataset....")

start = time.time()

X, Y = load_training_data(train_x_data, train_y_data, options=options, subcort_masks=None)
y = to_categorical(Y, num_classes=2)

print("\n data_shape: {}, {}".format(X.shape, y.shape))

if hostname.startswith("pandarus"):
    options['data_folder'] = '/host/pandarus/local_raid/ravnoor/data/'
elif hostname.startswith("hamlet"):
    options['data_folder'] = '/host/hamlet/local_raid/data/ravnoorX/data/'
else:
    options['data_folder'] = options['train_folder']

datapath = options['data_folder']+'hdf5/'+options['experiment']+'_LesionDilate_data.h5'
print("\n hdf5 dataset is being created: {}".format(datapath))

create_dataset(datapath, X, y)

end = time.time()
diff = end - start
print "time elapsed: ~ %i seconds" %(diff)
