#!/usr/bin/env python3

try:
    import h5py
except ImportError:
    raise ImportError('install h5py first: `pip install h5py --upgrade`')

import numpy as np

h5file = 'noel_FCDdata_N_patches_1000_patchsize_16_iso_fix.h5'
# h5file available from https://doi.org/10.5281/zenodo.3239446
f = h5py.File(h5file, 'r')

with h5py.File(h5file, "r") as f:
    X = f['data'][:].astype('f')
    y = f['labels'][:].astype('i8')

print(X.shape, y.shape)

print(np.histogram(y, bins=2))
