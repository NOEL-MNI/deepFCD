#!/usr/bin/env python3

import os, math, sys, time, fileinput, re
import subprocess
from subprocess import Popen, PIPE
# from local import *
import argparse
import numpy as np
import operator

import nibabel as nib
from nibabel.processing import resample_to_output as resample
from mo_dots import wrap, Data
import pandas as pd


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def get_nii_hdr_affine(t1w_fname):
    nifti = nib.load(t1w_fname)
    shape = nifti.get_data().shape
    header = nib.load(t1w_fname).header
    affine = header.get_qform()
    return nifti, header, affine, shape


def process_dense_CRF(id, args, root_dir='/tmp'):
    filespec = '_t1_final.nii.gz'
    t1w_fname = os.path.join(root_dir, id+filespec)
    print(id)
    start_time = time.time()
    case_id = id

    dst = os.path.join(args.basedir, case_id)
    outfile = os.path.join(dst, case_id+"_denseCrf3dSegmMap.nii.gz")

    _, header, affine, out_shape = get_nii_hdr_affine(t1w_fname) # load original input with header and affine

    print("save {}".format(case_id))
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)

    config='app/utils/dense3dCrf/config_densecrf.txt'
    start_time = time.time()
    denseCRF(case_id, t1w_fname, out_shape, config, dst, os.path.join(dst, case_id+"_vnet_maskpred.nii.gz"))
    elapsed_time = time.time() - start_time
    print("=*80")
    print("=> dense 3D-CRF inference time: {} seconds".format(round(elapsed_time,2)))
    print("=*80")


def denseCRF(id, t1, input_shape, config, out_dir, pred_labels):
    X, Y, Z = input_shape
    config_tmp = "/tmp/"+id+"_config_densecrf.txt"
    print(config_tmp)
    subprocess.call(["cp", "-f", config, config_tmp])
    find_str = ["<ID_PLACEHOLDER>", "<T1_FILE_PLACEHOLDER>", "<OUTDIR_PLACEHOLDER>", "<PRED_LABELS_PLACEHOLDER>", "<X_PLACEHOLDER>", "<Y_PLACEHOLDER>", "<Z_PLACEHOLDER>"]
    replace_str = [str(id), str(t1), str(out_dir), str(pred_labels), str(X), str(Y), str(Z)]

    for fs, rs in zip(find_str, replace_str):
        find_replace_re(config_tmp, fs, rs)
    subprocess.call(["app/utils/dense3dCrf/dense3DCrfInferenceOnNiis", "-c", config_tmp])


def find_replace_re(config_tmp, find_str, replace_str):
    with fileinput.FileInput(config_tmp, inplace=True, backup='.bak') as file:
        for line in file:
            print(re.sub(find_str, str(replace_str), line.rstrip(), flags=re.MULTILINE), end='\n')

args = Data()

args.basedir = 'app/weights'

scan = sys.argv[1]

process_dense_CRF(scan, args)
