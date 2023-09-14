#!/usr/bin/env python3
#%%
import logging
import multiprocessing
import os
import subprocess
import sys
import warnings
import bids
from bids import BIDSLayout

from tqdm.contrib.concurrent import process_map
from functools import partial
#%%
from config.experiment import options

warnings.filterwarnings("ignore")
import time

import numpy as np
import setproctitle as spt
from tqdm import tqdm

from utils.helpers import *

from preprocess_bids import preprocess_image

import argparse

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="{asctime} {levelname} {filename}:{lineno}: {message}",
)
os.environ["KERAS_BACKEND"] = "theano"

# configuration
parser = argparse.ArgumentParser(
                    prog='deepFCD',
                    description='deepFCD model',
                    epilog="I dare you to at the code!")
parser.add_argument('-bp','--bidspath')
parser.add_argument('-sp','--space')
# set to True or any non-zero value for brain extraction or skull-removal, False otherwise
parser.add_argument('-bm','--brainmask',action='store_true',default=False)
# co-register T1 and T2 images to MNI152 space and N3 correction before brain extraction (True/False)
parser.add_argument('-pp','--preprocess',action='store_true',default=False)
parser.add_argument('-o','--overwrite',action='store_true',default=False)
parser.add_argument('-dev','--device',default='cpu')
parser.add_argument('-s','--subjects', nargs='+', default=None)

args_ = parser.parse_args()

if not os.path.isabs(args_.bidspath):
    args_.bidspath = os.path.abspath(args_.bidspath)

print(args_.bidspath)
orig_ds = BIDSLayout(args_.bidspath, validate=False)
print(orig_ds)

if args_.subjects is None:
    subjects = orig_ds.get_subjects()
else:
    subjects = [s.replace('sub-','') for s in args_.subjects]
    print(subjects)
    
# GPU/CPU options
# cpu, cuda, cuda0, cuda1, or cudaX: flag using gpu 1 or 2
if args_.device.startswith("cuda1"):
    os.environ[
        "THEANO_FLAGS"
    ] = "mode=FAST_RUN,device=cuda1,floatX=float32,dnn.enabled=False"
elif args_.device.startswith("cpu"):
    cores = str(multiprocessing.cpu_count() // 2)
    var = os.getenv("OMP_NUM_THREADS", cores)
    try:
        logging.info("# of threads initialized: {}".format(int(var)))
    except ValueError:
        raise TypeError(
            "The environment variable OMP_NUM_THREADS"
            " should be a number, got '%s'." % var
        )
    # os.environ['openmp'] = 'True'
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,openmp=True,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32,dnn.enabled=False"
logging.info(os.environ["THEANO_FLAGS"])

from keras import backend as K
from keras.models import load_model

from models.noel_models_keras import *
from utils.base import *
from utils.metrics import *

outdir = os.path.join(os.path.dirname(args_.bidspath), "deepfcd")

cwd = os.path.realpath(os.path.dirname(__file__))
use_gpu = args_.device.startswith("cuda")
print(orig_ds)

outdir = os.path.join(os.path.dirname(args_.bidspath),'deepfcd')
os.makedirs(outdir,exist_ok=True)
with open(os.path.join(outdir,'dataset_description.json'),'w') as f:
    f.write('{"Name": "fov","BIDSVersion": "1.7.0","DatasetType": "derivative","PipelineDescription": {"Name": "antsRegistration"}}')
   
if args_.brainmask:
    #multiproc
    t1w_paths = []
    flair_paths = []
    fullids=[]
    for s in subjects:
        t1w_paths.append(os.path.basename(orig_ds.get(subject=s,space=args_.space,suffix='T1w')[0].path))
        flair_paths.append(orig_ds.get(subject=s,space=args_.space,suffix='FLAIR')[0].path)
        fullids.append(f"sub-{s}")

    process_map(partial(preprocess_image,indir_=args_.bidspath,outdir_=outdir,preprocess=args_.preprocess, use_gpu=use_gpu),fullids,t1w_paths,flair_paths)

    # for s in subjects:
    #     t1w_path = orig_ds.get(subject=s,space=args_.space,suffix='T1w')[0].path
    #     flair_path = orig_ds.get(subject=s,space=args_.space,suffix='FLAIR')[0].path
    #     preprocess_image(id_=f"sub-{s}", t1_fname=os.path.basename(t1w_path), t2_fname=os.path.basename(flair_path), indir_=args_.bidspath,outdir_=outdir,preprocess=args_.preprocess, use_gpu=use_gpu)

else:
    logging.info(
        "Skipping image preprocessing and brain masking, presumably images are co-registered, bias-corrected, and skull-stripped"
    )

proc_ds = BIDSLayout(outdir, validate=False)
if args_.subjects is None:
    subjects = proc_ds.get_subjects()
else:
    subjects = [s.replace('sub-','') for s in args_.subjects]
    print(subjects)
    
print(proc_ds)
#%%
# sys.exit(0)
#%%
# deepFCD configuration
K.set_image_dim_ordering("th")
K.set_image_data_format("channels_first")  # TH dimension ordering in this code

options["parallel_gpu"] = False
modalities = ["T1", "FLAIR"]
x_names = options["x_names"]

# seed = options['seed']
options["dropout_mc"] = True # TODO was True
options["batch_size"] = 350000
options["mini_batch_size"] = 2048
options["load_checkpoint_1"] = True
options["load_checkpoint_2"] = True

# trained model weights based on 148 histologically-verified FCD subjects
options["test_folder"] = outdir
options["weight_paths"] = os.path.join(cwd, "weights")
options["experiment"] = "noel_deepFCD_dropoutMC"
logging.info("experiment: {}".format(options["experiment"]))
spt.setproctitle(options["experiment"])

#%%
# sys.exit(0)
#%%
# --------------------------------------------------
# initialize the CNN
# --------------------------------------------------
# initialize empty model
model = None
# initialize the CNN architecture
model = off_the_shelf_model(options)

load_weights = os.path.join(
    options["weight_paths"], "noel_deepFCD_dropoutMC_model_1.h5"
)
logging.info(
    "loading DNN1, model[0]: {} exists".format(load_weights)
) if os.path.isfile(load_weights) else sys.exit(
    "model[0]: {} doesn't exist".format(load_weights)
)
model[0] = load_model(load_weights)

load_weights = os.path.join(
    options["weight_paths"], "noel_deepFCD_dropoutMC_model_2.h5"
)
logging.info(
    "loading DNN2, model[1]: {} exists".format(load_weights)
) if os.path.isfile(load_weights) else sys.exit(
    "model[1]: {} doesn't exist".format(load_weights)
)
model[1] = load_model(load_weights)
logging.info(model[1].summary())

# --------------------------------------------------
# test the cascaded model
# --------------------------------------------------
# test_list = ['mcd_0468_1']
# sys.exit(0)
for s in tqdm(subjects, desc="serving predictions using the trained model", colour="blue"):
    fullid = f"sub-{s}"
    options['fullid'] = fullid
    # t1_file = ds.get(subject=s,space=args_.space,suffix='T1w')[0].path
    'label-brain_FLAIR.nii.gz'
    'label-brain_T1w.nii.gz'
    
    t1_file = proc_ds.get(subject=s, space='MNI152NLin2009aSym', label='brain', suffix='T1w')[0].path
    t2_file = proc_ds.get(subject=s, space='MNI152NLin2009aSym', label='brain', suffix='FLAIR')[0].path
    orig_bidsfiles = [
        orig_ds.get(subject=s,space=args_.space,suffix='T1w')[0],
        orig_ds.get(subject=s,space=args_.space,suffix='FLAIR')[0] 
        ]
    orig_files = [bf.path for bf in orig_bidsfiles]
    
    t1_transform = proc_ds.get(subject=s, extension='mat', suffix='T1w')[0].path
    t2_transform = proc_ds.get(subject=s, extension='mat', suffix='FLAIR')[0].path

    files = [t1_file, t2_file]

    transform_files = [t1_transform, t2_transform]

    test_data = {}
    test_data = {fullid: {
            m: f for m, f in zip(modalities, files) # TOCHECK
        }
    }
    test_transforms = {fullid: {m: n for m, n in zip(modalities, transform_files)}}
    # test_data = {f: {m: os.path.join(options['test_folder'], f, n) for m, n in zip(modalities, files)} for f in test_list}


    t_data = {}
    t_data[fullid] = test_data[fullid]
    transforms = {}
    transforms[fullid] = test_transforms[fullid]

    options["pred_folder"] = os.path.join(
        options["test_folder"], fullid, options["experiment"]
    )
    os.makedirs(options["pred_folder"], exist_ok=True)

    pred_mean_fname = os.path.join(options["pred_folder"], f"{fullid}_space-MNI152NLin2009aSym_acq-{options['experiment']}Mean1_pred.nii.gz")
    pred_var_fname = os.path.join(options["pred_folder"], f"{fullid}_space-MNI152NLin2009aSym_acq-{options['experiment']}Var1_pred.nii.gz")

    if np.logical_and(os.path.isfile(pred_mean_fname), os.path.isfile(pred_var_fname)):
        logging.info("prediction for {} already exists".format(fullid))
        if not args_.overwrite:
            transform_img(pred_mean_fname,bids.layout.parse_file_entities(pred_mean_fname),orig_files[0],transform_files[0],targetspace=orig_bidsfiles[0].entities['space'],invert=True)
            transform_img(pred_var_fname,bids.layout.parse_file_entities(pred_var_fname),orig_files[0],transform_files[0],targetspace=orig_bidsfiles[0].entities['space'],invert=True)
            continue
        else:
            logging.info("overwriting...")

    options["test_scan"] = fullid

    start = time.time()
    logging.info("\n")
    logging.info("-" * 70)
    logging.info("testing the model for scan: {}".format(fullid))
    logging.info("-" * 70)

    # if transform(s) do not exist (i.e., no preprocessing done), then skip (see base.py#L412)
    if not any([os.path.exists(transforms[fullid]["T1"]), os.path.exists(transforms[fullid]["FLAIR"])]):
        transforms = None

    outputs = test_model(
        model,
        t_data,
        options,
        performance=True,
        uncertainty=True,
        transforms=transforms,
        orig_files=orig_files,
        invert_xfrm=True,
    )
    #TODO loop over transforms, for now just use first
    for k,v in outputs.items():
        transform_img(v,bids.layout.parse_file_entities(v),orig_files[0],transform_files[0],targetspace=orig_bidsfiles[0].entities['space'],invert=True)

    end = time.time()
    diff = (end - start) // 60
    logging.info("-" * 70)
    logging.info("time elapsed: ~ {} minutes".format(diff))
    logging.info("-" * 70)
