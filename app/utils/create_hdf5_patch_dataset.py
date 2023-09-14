#!/usr/bin/env python3

import os
try:
    import h5py
except ImportError:
    raise ImportError('install h5py first: `pip install h5py --upgrade`')

import numpy as np

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
print(os.environ["THEANO_FLAGS"])

import time
from h5data import load_training_data, create_dataset

# set configuration parameters
options = {}
options["n_patches"] = 1500
options["seed"] = 666
options["modalities"] = ["T1", "FLAIR"]
options["x_names"] = ["_t1.nii.gz", "_flair.nii.gz"]
options["y_names"] = ["_lesion.nii.gz"]
options["submask_names"] = ["subcorticalMask_final_negative.nii.gz"]
options["patch_size"] = (16, 16, 16)

options["thr"] = 0.1
options["min_th"] = options["thr"]

# randomize training features before fitting the model.
options["randomize_train"] = True

modalities = options["modalities"]
x_names = options["x_names"]
y_names = options["y_names"]

seed = options["seed"]
print("seed: {}".format(seed))
# Select an experiment name to store net weights and segmentation masks
options["experiment"] = "noel_FCDdata"

options["model_dir"] = "./weights"  # weights/noel_dropoutMC_model_{1,2}.h5
options["train_folder"] = "/host/hamlet/local_raid/data/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/data/"
# options["data_folder"] = "/host/hamlet/local_raid/data/ravnoorX/data/noel_hdf5"
options["data_folder"] = "/tmp/noel_hdf5"

list_of_train_scans = os.listdir(options["train_folder"] + "brain")
include_train = list(set(list_of_train_scans))

modality = [x.lower() for x in modalities]

for m in modality:
    include_train = [f.replace("_" + m + ".nii.gz", "") for f in include_train]
include_train = list(set(include_train))

print("training dataset size: {}".format(len(include_train)))

modality = [x.upper() for x in modality]

for scan in include_train:
    # load paths to all the data
    train_x_data = {
        f: {
            m: os.path.join(options["train_folder"], "brain", f + n)
            for m, n in zip(modality, x_names)
        }
        for f in include_train
    }
    train_y_data = {
        f: os.path.join(options["train_folder"], "lesion_labels", f + y_names[0])
        for f in include_train
    }

print(
    "\nconverting 3D MRI to patch-based dataset with {} patches of size: {}".format(
        options["n_patches"], options["patch_size"]
    )
)

start = time.time()

X, y = load_training_data(
    train_x_data, train_y_data, options=options, subcort_masks=None
)
# y = to_categorical(Y, num_classes=2)

print("\ndata_shape: {}, {}".format(X.shape, y.shape))

h5_fname = (
    options["experiment"]
    + "_N_patches_"
    + str(options["n_patches"])
    + "_patchsize_"
    + str(options["patch_size"][0])
    + "_iso_fix.h5"
)

print(np.histogram(y, bins=2))

datapath = os.path.join(options["data_folder"], h5_fname)
print("\nhdf5 dataset is being created: {}".format(datapath))

create_dataset(datapath, X, y)

end = time.time()
diff = end - start
print("time elapsed: ~ {} minutes".format(diff // 60))

# validate the newly created dataset
print("\nhdf5 dataset is being loaded: {}".format(datapath))

# sample hdf5 dataset available from https://doi.org/10.5281/zenodo.3239446
with h5py.File(datapath, "r") as f:
    X = f['data'][:].astype('f')
    y = f['labels'][:].astype('i')

# output the shape of the patches and labels
print(X.shape, y.shape)

# should output equal number of positive and negative examples (0/1)
print(np.histogram(y, bins=2))