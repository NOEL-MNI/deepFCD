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
    options['cuda'] = sys.argv[2] # flag using gpu 1 or 2
    if options['cuda'].startswith('cuda1'):
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1,floatX=float32"
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

test_site = sys.argv[1]

options['parallel_gpu'] = False
modalities = options['modalities']
x_names = options['x_names']
options['y_names'] = ['_lesion.nii.gz']
y_names = options['y_names']

# seed = options['seed']

options['dropout_mc'] = True
options['batch_size'] = 350000
options['mini_batch_size'] = 2048

# Select an experiment name to store net weights and segmentation masks
options['experiment'] = 'exp_dropoutMC_' + test_site

print("experiment: {}".format(options['experiment']))
spt.setproctitle(options['experiment'])

options['model_dir'] = '/host/silius/local_raid/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/models'
if hostname.startswith("pandarus"):
    options['hdf5_data_dir'] = '/host/pandarus/local_raid/ravnoor/data/hdf5'
elif hostname.startswith("hamlet"):
    options['hdf5_data_dir'] = '/host/hamlet/local_raid/data/ravnoorX/data/hdf5'
else:
    options['hdf5_data_dir'] = '/host/silius/local_raid/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/data/hdf5'

sensitivity = 0
perf = {}

options['train_folder'] = '/host/silius/local_raid/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/data/'
options['test_folder'] = '/host/silius/local_raid/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/data/'

datafile = '/host/silius/local_raid/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/src/data_site_scanner_labels.xlsx'

train, test, folds = partition_leave_one_site_out(datafile = datafile, test_site = test_site)

options['load_checkpoint_1'] = False
options['load_checkpoint_2'] = False
# options['continue_training_2'] = True
# options['initial_epoch_2'] = 69

train_list, test_list = [], []
train_data, test_data = {}, {}

for i in train:
    train_list.append(i)

for i in test:
    test_list.append(i)

print(test_list)

train_data = {f: {m: os.path.join(options['train_folder'], 'brain', f+n) for m, n in zip(modalities, x_names)} for f in train_list}
train_labels = {f: os.path.join(options['train_folder'], 'lesion_labels', f+y_names[0]) for f in train_list}

test_data = {f: {m: os.path.join(options['train_folder'], 'brain', f+n) for m, n in zip(modalities, x_names)} for f in test_list}
test_labels = {f: os.path.join(options['train_folder'], 'lesion_labels', f+y_names[0]) for f in test_list}

# --------------------------------------------------
# initialize the CNN
# --------------------------------------------------
options['weight_paths'] = os.path.join(options['model_dir'], options['experiment'])

model = None  # Clearing the CNN.

model = off_the_shelf_model(options)

print(model[0].summary())

try:
    os.mkdir(options['weight_paths'])
except:
    print('{} exists'.format(options['weight_paths']))

start = time.time()

model = train_model(model, train_data, train_labels, options=options)

if not options['load_checkpoint_2']:
    print("Saving fold split info")
    f = open((options['weight_paths'] + '/' + '/fold_info.txt'), "w")
    f.write("training list: %s \n \n" % (folds[test_site]['train_pids']) )
    f.write("test list: %s \n \n" % (folds[test_site]['test_pids']) )
    f.close()

    print("Saving config")
    opts_json = json.dumps(options, indent=4, sort_keys=True)
    f = open(options['weight_paths'] + '/' + options['experiment'] + '_config.json', "w")
    f.write(opts_json)
    f.close()

    print("Saving trained models")
    model[0].save(options['weight_paths'] + '/' + options['experiment'] + '_model_1.h5')
    model[1].save(options['weight_paths'] + '/' + options['experiment'] + '_model_2.h5')

end = time.time()
diff = end - start
print("training or loading model time elapsed: ~ {} seconds".format(diff))


# --------------------------------------------------
# test the cascaded model
# --------------------------------------------------

# for scan in test_list:
for _, scan in enumerate(tqdm(test_list, desc='serving predictions using the trained model')):

    t_data = {}
    t_data[scan] = test_data[scan]

    # print(t_data)

    # test_folder = os.path.join(options['test_folder'], 'predictions')
    test_folder = '/host/silius/local_raid/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/data/predictions'
    options['pred_folder'] = os.path.join(test_folder, options['experiment'], scan)

    pred_mean_fname = os.path.join(options['pred_folder'], options['experiment'] + '_prob_mean_1.nii.gz')
    pred_var_fname = os.path.join(options['pred_folder'], options['experiment'] + '_prob_var_1.nii.gz')

    if np.logical_and(os.path.isfile(pred_mean_fname), os.path.isfile(pred_var_fname)):
        print("prediction for {} already exists".format(scan))
        continue

    if not os.path.exists(options['pred_folder']):
        # os.path.join(test_folder, options['experiment'])
        os.mkdir(options['pred_folder'])
        # os.mkdir(os.path.join(test_folder, options['experiment']))
        # os.mkdir(os.path.join(test_folder, options['experiment'], scan))

    options['test_name'] = scan + '_' + options['experiment'] + '.nii.gz'
    options['test_scan'] = scan

    start = time.time()

    print('\n')
    print('-'*80)
    print("testing the model for scan: {} ".format(scan))
    print('-'*80)

    test0, test1, test2, lpred, count = test_model(model, t_data, options)   # test0: prediction/stage1 | test1: pred/stage2 | test2: morphological processing + clustered | lpred: predicted label only | count: # false positives

    y_true = True
    if y_true:
        label = np.asarray(load_nii(test_labels[scan]).get_data())
        # print("label_shape: {}, label_unique: {}".format(label.shape, np.unique(label)))
        lesion_pred = extract_lesional_clus(label, test1, scan, options)

        print("number of non-zero voxels after CNN#1: {}".format(np.count_nonzero(test0)))
        print("number of non-zero voxels after CNN#2: {}".format(np.count_nonzero(test1)))
        print("number of non-zero voxels after size thresholding (> 20 voxels): {}".format(np.count_nonzero(test2)))

        print('-'*80)
        print("computing performance metrics")
        print('-'*80)

        perf = performancer(perf, scan, test2, label, lesion_pred, count)

        print('-'*80)
        end = time.time()
        diff = end - start
        print("="*80)
        print("time elapsed: ~ {} seconds".format(diff))
        print("="*80)

        if perf[scan]['sensitivity'] != 0:
            sensitivity += 1

if y_true:
    csv_name = test_folder+'/'+'results_tbin_'+str(options['t_bin'])+'_lmin_'+str(options['l_min'])+'_'+str(options['experiment'])+'.csv'
    df = pd.DataFrame(perf)
    df = df.transpose()
    df.to_csv(csv_name)
