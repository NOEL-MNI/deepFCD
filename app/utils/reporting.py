#!/usr/bin/env python
# coding: utf-8

"""Rank clusters based on probability/size thresholding and uncertainty,
    and prints output

Usage:
    conda activate deepFCD
    python3 reporting.py ${PATIENT_ID} ${IO_DIRECTORY} [--p_thr PROB_THRESH] [--c_thr CLUSTER_SIZE]
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
from nibabel import load as load_nii
from sklearn import preprocessing
from tabulate import tabulate

from atlasreader.atlasreader import read_atlas_peak
from confidence import extractLesionCluster

# parse command line arguments
parser = argparse.ArgumentParser(description='Rank clusters based on probability/size thresholding and uncertainty')
parser.add_argument('patient_id', help='Patient ID')
parser.add_argument('io_directory', help='Input/Output directory')
parser.add_argument('--p_thr', '--prob_thr', type=float, default=0.7, help='Probability threshold (default: 0.7)')
parser.add_argument('--c_thr', '--clus_thr', type=int, default=300, help='Cluster size threshold (default: 300)')

args = parser.parse_args()

scan = args.patient_id
options = {}
options["data_folder"] = os.path.join(args.io_directory, scan, "noel_deepFCD_dropoutMC")

modality = [
    "_noel_deepFCD_dropoutMC_prob_mean_1.nii.gz",
    "_noel_deepFCD_dropoutMC_prob_var_1.nii.gz",
]
data_bayes, data_bayes_var = {}, {}

cwd = os.path.realpath(os.path.dirname(__file__))

# mask to exclude all subcortical findings
options["submask"] = os.path.join(cwd, "../templates", "subcortical_mask_v3.nii.gz")

# load paths to all the data
data_bayes[scan] = os.path.join(options["data_folder"], scan + str(modality[0]))
data_bayes_var[scan] = os.path.join(options["data_folder"], scan + str(modality[1]))

ea = load_nii(data_bayes[scan]).get_fdata()
ea_var = load_nii(data_bayes_var[scan]).get_fdata()

options["header"] = load_nii(data_bayes[scan]).header

options["t_bin"] = args.p_thr  # probability threshold from CLI args
options["l_min"] = args.c_thr  # cluster size threshold from CLI args

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
results["confidence"] = np.round(
    100.0 * min_max_scaler.fit_transform(invert_var.values.reshape(-1, 1)), 1
)
ranked_results = results.sort_values("rank")
ranked_results.reset_index(inplace=True)

labels = []
for N in np.arange(0, len(ranked_results.coords)):
    label = read_atlas_peak(
        atlastype="harvard_oxford", coordinate=ranked_results.coords[N], prob_thresh=5
    )
    labels.append(label[0][1])

ranked_results["region"] = labels

ranked_results["probability"] = ranked_results["probability"].apply(
    lambda x: int(round(x * 100))
)
ranked_results["confidence"] = ranked_results["confidence"].apply(
    lambda x: int(round(x))
)

# remove spaces and square brackets from coordinates
ranked_results["coords"] = ranked_results["coords"].apply(
    lambda x: str(x).replace(" ", "").replace("[", "").replace("]", "")
)

cols_pref = ["rank", "region", "coords", "probability", "confidence"]
print(
    tabulate(
        ranked_results[cols_pref], headers="keys", tablefmt="github", showindex=False
    )
)

p_thr = options["t_bin"]
c_thr = options["l_min"]
csv_file = os.path.join(options["data_folder"], f"ranked_results_{scan}-pthr_{p_thr}-cthr_{c_thr}.csv")
ranked_results[cols_pref].to_csv(csv_file, index=False)
# ranked_results[cols_pref].to_csv(f"ranked_results_{scan}.csv", index=False)

nib.save(out_scan, os.path.join(options["data_folder"], scan + f"_clusters-pthr_{p_thr}-cthr_{c_thr}.nii.gz"))
