#!/usr/bin/env python
# coding: utf-8
# conda activate py3keras
# usage: python3 postprocessing.py FCD_001 /var/data/FCD_001/noel_deepFCD_dropoutMC/

import os
import sys

import nibabel as nib
import numpy as np
from nibabel import load as load_nii

from atlasreader.atlasreader import read_atlas_peak
from confidence import extractLesionCluster

scan = sys.argv[1]
options = {}
options["data_folder"] = os.path.join(sys.argv[2], scan, "noel_deepFCD_dropoutMC")
# modality = ["_deep_prob_1.nii.gz", "_deep_var_1.nii.gz"]
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

options["t_bin"] = 0.6
options["l_min"] = 150

scan_keys = []

for k in data_bayes.keys():
    scan_keys.append(k)

results = {}
output_scan, results = extractLesionCluster(scan, ea, ea_var, options)

header = load_nii(data_bayes[scan]).header
affine = header.get_qform()
out_scan = nib.Nifti1Image(output_scan, affine=affine, header=header)

results.sort_values("rank")
# results["var"].max()
# results["var2"] = 1 / results["var"]
# results["var3"] = results["var2"] / results["var2"].max()
results["confidence"] = (1 / results["var"]) / results["var"].max()
ranked_results = results.sort_values("rank")
ranked_results.reset_index(inplace=True)

labels = []
for N in np.arange(0,len(ranked_results.coords)):
    label = read_atlas_peak(atlastype='harvard_oxford', coordinate=ranked_results.coords[N], prob_thresh=5)
    for _, (perc, l) in label[0]:
        print(perc, l)
    labels.append(label[0])
    # print(label)

ranked_results['label'] = labels
print(ranked_results)
# print(results.sort_values("rank").coords)

# simulation = pd.DataFrame(np.random.random(20) * 0.3, columns=["var"])

# simulation["var2"] = 1 / simulation["var"]
# simulation["var3"] = simulation["var2"] / simulation["var2"].max()
# simulation["var4"] = (1 / simulation["var"]) / simulation["var"].max()

# simulation.sort_values("var")

# clust_frame, _ = get_statmap_info(data_bayes[scan], cluster_extent=options["l_min"], direction='both', atlas='default', 
#                                     voxel_thresh=options["t_bin"], prob_thresh=options["t_bin"], min_distance=None)
# print(clust_frame)
# print(peaks_frame)

# read_atlas_cluster(atlastype, cluster, affine, prob_thresh=options["t_bin"]+0.1)
