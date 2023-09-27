#!/usr/bin/env python
# coding: utf-8

'''Rank clusters based on probability/size thresholding and uncertainty, 
    and prints output

Usage:
    conda activate deepFCD
    python3 reporting.py ${PATIENT_ID} ${IO_DIRECTORY}
'''

import os
import sys

import nibabel as nib
import numpy as np
from nibabel import load as load_nii
from sklearn import preprocessing
from tabulate import tabulate

from atlasreader.atlasreader import read_atlas_peak
from confidence import extractLesionCluster

scan = sys.argv[1]
options = {}
options["data_folder"] = os.path.join(sys.argv[2], scan, "noel_deepFCD_dropoutMC")

modality = [
    "_noel_deepFCD_dropoutMC_prob_mean_1.nii.gz",
    "_noel_deepFCD_dropoutMC_prob_var_1.nii.gz",
]
data_bayes, data_bayes_var = {}, {}

cwd = os.path.realpath(os.path.dirname(__file__))
# mask to exclude all subcortical findings
options['submask'] = os.path.join(cwd, '../templates', 'subcortical_mask_v3.nii.gz')

# load paths to all the data
data_bayes[scan] = os.path.join(options["data_folder"], scan + str(modality[0]))
data_bayes_var[scan] = os.path.join(options["data_folder"], scan + str(modality[1]))

ea = load_nii(data_bayes[scan]).get_fdata()
ea_var = load_nii(data_bayes_var[scan]).get_fdata()

options["header"] = load_nii(data_bayes[scan]).header

options["t_bin"] = 0.6  # probability threshold
options["l_min"] = 150  # cluster size threshold

scan_keys = []

for k in data_bayes.keys():
    scan_keys.append(k)

results = {}
output_scan, results = extractLesionCluster(scan, ea, ea_var, options)

header = load_nii(data_bayes[scan]).header
affine = header.get_qform()
out_scan = nib.Nifti1Image(output_scan, affine=affine, header=header)

results.sort_values("rank")
min_max_scaler = preprocessing.MinMaxScaler()
invert_var = 1 / results["var"]
results["confidence"] = np.round(100.0 * min_max_scaler.fit_transform(invert_var.values.reshape(-1, 1)), 1)
ranked_results = results.sort_values("rank")
ranked_results.reset_index(inplace=True)

labels = []
for N in np.arange(0,len(ranked_results.coords)):
    label = read_atlas_peak(atlastype='harvard_oxford', coordinate=ranked_results.coords[N], prob_thresh=5)
    # for (perc, l) in label[0]:
    #     print(perc, l)
    labels.append(label[0])
    # print(label)

ranked_results['label'] = labels
print(tabulate(ranked_results, headers = 'keys', tablefmt = 'simple'))