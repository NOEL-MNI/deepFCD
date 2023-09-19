#!/usr/bin/env python3

import logging
import multiprocessing
import os
import subprocess
import sys
import warnings

from mo_dots import Data

from config.experiment import options

warnings.filterwarnings("ignore")
import time

import numpy as np
import setproctitle as spt
from tqdm import tqdm

from utils.helpers import *

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="{asctime} {levelname} {filename}:{lineno}: {message}",
)

os.environ["KERAS_BACKEND"] = "theano"

# GPU/CPU options
options["cuda"] = sys.argv[5]
# cpu, cuda, cuda0, cuda1, or cudaX: flag using gpu 1 or 2
if options["cuda"].startswith("cuda1"):
    os.environ[
        "THEANO_FLAGS"
    ] = "mode=FAST_RUN,device=cuda1,floatX=float32,dnn.enabled=False"
elif options["cuda"].startswith("cpu"):
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

# configuration
args = Data()
args.id = sys.argv[1]
args.t1_fname = sys.argv[2]
args.t2_fname = sys.argv[3]
args.dir = sys.argv[4]
if not os.path.isabs(args.dir):
    args.dir = os.path.abspath(args.dir)

args.brain_masking = int(sys.argv[6])
# set to True or any non-zero value for brain extraction or skull-removal, False otherwise
args.preprocess = int(sys.argv[7])
# co-register T1 and T2 images to MNI152 space and N3 correction before brain extraction (True/False)
args.outdir = os.path.join(args.dir, args.id)

args.t1 = os.path.join(args.outdir, args.t1_fname)
args.t2 = os.path.join(args.outdir, args.t2_fname)

args.t1_orig, args.t2_orig = args.t1, args.t2

cwd = os.path.realpath(os.path.dirname(__file__))

if bool(args.brain_masking):
    if options["cuda"].startswith("cuda"):
        args.use_gpu = True
    else:
        args.use_gpu = False
    # MRI pre-processing configuration
    args.output_suffix = "_brain_final.nii.gz"

    preprocess_sh = os.path.join(cwd, "preprocess.sh")
    subprocess.check_call(
        [
            preprocess_sh,
            args.id,
            args.t1_fname,
            args.t2_fname,
            args.dir,
            bool2str(args.preprocess),
            bool2str(args.use_gpu),
        ]
    )

    args.t1 = os.path.join(args.outdir, args.id + "_t1" + args.output_suffix)
    args.t2 = os.path.join(args.outdir, args.id + "_t2" + args.output_suffix)
else:
    logging.info(
        "Skipping image preprocessing and brain masking, presumably images are co-registered, bias-corrected, and skull-stripped"
    )

if os.environ.get("CI_TESTING") is not None:
    options["CI_TESTING_GT"] = os.environ.get("CI_TESTING_GT")
    print("CI environment initialized: {}".format(options["CI_TESTING_GT"]))
    mask = ants.image_read(options["CI_TESTING_GT"])
    t1, t2 = ants.image_read(args.t1), ants.image_read(args.t2)
    ants.mask_image(t1, mask, level=1, binarize=False).to_filename(args.t1)
    ants.mask_image(t2, mask, level=1, binarize=False).to_filename(args.t2)

# deepFCD configuration
K.set_image_dim_ordering("th")
K.set_image_data_format("channels_first")  # TH dimension ordering in this code

options["parallel_gpu"] = False
modalities = ["T1", "FLAIR"]
x_names = options["x_names"]

# seed = options['seed']
options["dropout_mc"] = True
options["batch_size"] = 350000
options["mini_batch_size"] = 2048
options["load_checkpoint_1"] = True
options["load_checkpoint_2"] = True

# trained model weights based on 148 histologically-verified FCD subjects
options["test_folder"] = args.dir
options["weight_paths"] = os.path.join(cwd, "weights")
options["experiment"] = "noel_deepFCD_dropoutMC"
logging.info("experiment: {}".format(options["experiment"]))
spt.setproctitle(options["experiment"])

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
test_list = [args.id]
# t1_file = sys.argv[3]
# t2_file = sys.argv[4]
t1_file = args.t1
t2_file = args.t2

t1_transform = os.path.join(
    args.outdir, "transforms", args.id + "_t1-native-to-MNI152.mat"
)
t2_transform = os.path.join(
    args.outdir, "transforms", args.id + "_t2-native-to-MNI152.mat"
)

files = [args.t1, args.t2]

orig_files = {"T1": args.t1_orig, "FLAIR": args.t2_orig}

transform_files = [t1_transform, t2_transform]
# files = {}
# files['T1'], files['FLAIR'] = str(t1_file), t2_file
test_data = {}
# test_data = {f: {m: os.path.join(tfolder, f, m+'_stripped.nii.gz') for m in modalities} for f in test_list}
test_data = {
    f: {
        m: os.path.join(options["test_folder"], f, n) for m, n in zip(modalities, files)
    }
    for f in test_list
}
test_transforms = {
    f: {m: n for m, n in zip(modalities, transform_files)} for f in test_list
}
# test_data = {f: {m: os.path.join(options['test_folder'], f, n) for m, n in zip(modalities, files)} for f in test_list}

for _, scan in enumerate(
    tqdm(test_list, desc="serving predictions using the trained model", colour="blue")
):
    t_data = {}
    t_data[scan] = test_data[scan]
    transforms = {}
    transforms[scan] = test_transforms[scan]

    options["pred_folder"] = os.path.join(
        options["test_folder"], scan, options["experiment"]
    )
    if not os.path.exists(options["pred_folder"]):
        os.mkdir(options["pred_folder"])

    pred_mean_fname = os.path.join(options["pred_folder"], scan + "_prob_mean_1.nii.gz")
    pred_var_fname = os.path.join(options["pred_folder"], scan + "_prob_var_1.nii.gz")

    if np.logical_and(os.path.isfile(pred_mean_fname), os.path.isfile(pred_var_fname)):
        logging.info("prediction for {} already exists".format(scan))
        continue

    options["test_scan"] = scan

    start = time.time()
    logging.info("\n")
    logging.info("-" * 70)
    logging.info("testing the model for scan: {}".format(scan))
    logging.info("-" * 70)

    # if transform(s) do not exist (i.e., no preprocessing done), then skip (see base.py#L412)
    if not any([os.path.exists(transforms[scan]["T1"]), os.path.exists(transforms[scan]["FLAIR"])]):
        transforms = None

    test_model(
        model,
        t_data,
        options,
        transforms=transforms,
        orig_files=orig_files,
        invert_xfrm=True,
    )

    end = time.time()
    diff = (end - start) // 60
    logging.info("-" * 70)
    logging.info("time elapsed: ~ {} minutes".format(diff))
    logging.info("-" * 70)
