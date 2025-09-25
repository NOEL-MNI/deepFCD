#!/usr/bin/env python
# coding: utf-8

"""
Rank clusters based on probability/size thresholding and uncertainty,
    and prints output - BIDS compliant version

Usage:
    conda activate deepFCD
    python3 reporting_bids.py ${SUBJECT_ID} ${BIDS_DERIVATIVES_DIR} [--session SESSION] [--p_thr PROB_THRESH] [--c_thr CLUSTER_SIZE] [--space SPACE]
"""

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import load as load_nii
from sklearn import preprocessing
from tabulate import tabulate
from bids import BIDSLayout

from atlasreader.atlasreader import read_atlas_peak
from confidence import extractLesionCluster

# Parse command line arguments
parser = argparse.ArgumentParser(description='Rank clusters based on probability/size thresholding and uncertainty - BIDS compliant')
parser.add_argument('subject_id', help='Subject ID (e.g., sub-PX034 or PX034)')
parser.add_argument('derivatives_dir', help='Path to BIDS derivatives directory (e.g., /path/to/bids/derivatives/deepFCD)')
parser.add_argument('--session', help='Session ID (e.g., ses-01 or 01)')
parser.add_argument('--space', default='orig', help='Space to use for analysis (default: orig, can be MNI152)')
parser.add_argument('--p_thr', '--prob_thr', type=float, default=0.7, help='Probability threshold (default: 0.7)')
parser.add_argument('--c_thr', '--clus_thr', type=int, default=300, help='Cluster size threshold (default: 300)')

args = parser.parse_args()

# Normalize subject ID
subject_id = args.subject_id
if not subject_id.startswith('sub-'):
    subject_id = f'sub-{subject_id}'

# Normalize session ID if provided
session_id = None
if args.session:
    session_id = args.session
    if not session_id.startswith('ses-'):
        session_id = f'ses-{session_id}'
# Initialize BIDS layout for derivatives
try:
    layout = BIDSLayout(args.derivatives_dir, validate=False)
except Exception as e:
    print(f"Error loading BIDS layout from {args.derivatives_dir}: {e}")
    sys.exit(1)

# Build full subject identifier
if session_id:
    fullid = f"{subject_id}_{session_id}"
else:
    fullid = subject_id

# Find prediction files using BIDS layout with custom entities
# Since 'stat' is not a standard BIDS entity, we use invalid_filters='allow'
query_params = {
    'subject': subject_id.replace('sub-', ''),
    'desc': 'deepFCD',
    'suffix': 'probseg',
    'extension': '.nii.gz'
}

if session_id:
    query_params['session'] = session_id.replace('ses-', '')

if args.space != 'orig':
    query_params['space'] = args.space

# First try to find all deepFCD files, then filter by stat
all_files = layout.get(**query_params, invalid_filters='allow')

# Filter for mean1 and var1 files manually since stat is not standard BIDS
mean_files = [f for f in all_files if 'stat-mean1' in f.filename]
var_files = [f for f in all_files if 'stat-var1' in f.filename]

if not mean_files:
    print(f"Error: No mean1 prediction files found for {fullid} in space-{args.space}")
    print(f"Search parameters: {query_params}")
    # List available files for debugging
    print(f"All deepFCD files found: {[f.filename for f in all_files]}")
    if all_files:
        print("Available files by path:")
        for f in all_files:
            print(f"  {f.path}")
    else:
        print("No deepFCD files found at all. Check that inference has been run.")
    sys.exit(1)

if not var_files:
    print(f"Error: No var1 prediction files found for {fullid} in space-{args.space}")
    print(f"Available mean files: {[f.filename for f in mean_files]}")
    sys.exit(1)

# Use the first matching files
mean_file = mean_files[0].path
var_file = var_files[0].path

print(f"Using mean file: {mean_file}")
print(f"Using variance file: {var_file}")

# Load the data
ea = load_nii(mean_file).get_fdata()
ea_var = load_nii(var_file).get_fdata()

# Setup options
options = {}
options["header"] = load_nii(mean_file).header

# Set data folder to the output directory (same as where input files are)
options["data_folder"] = str(Path(mean_file).parent)

cwd = os.path.realpath(os.path.dirname(__file__))
# mask to exclude all subcortical findings
options["submask"] = os.path.join(cwd, "../templates", "subcortical_mask_v3.nii.gz")

options["t_bin"] = args.p_thr  # probability threshold from CLI args
options["l_min"] = args.c_thr  # cluster size threshold from CLI args

# Extract cluster information
results = {}
output_scan, results = extractLesionCluster(fullid, ea, ea_var, options)

header = load_nii(mean_file).header
affine = header.get_qform()
out_scan = nib.Nifti1Image(output_scan, affine=affine, header=header)

# Check if results is empty
if results.empty:
    print(f"No clusters found for {fullid} with p_thr={args.p_thr} and c_thr={args.c_thr}")
    sys.exit(0)

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

# Save results in BIDS-compliant format
p_thr = options["t_bin"]
c_thr = options["l_min"]

# Create output directory following BIDS structure
output_dir = Path(mean_file).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Generate BIDS-compliant output filenames
space_suffix = f"_space-{args.space}" if args.space != 'orig' else "_space-orig"
session_suffix = f"_{session_id}" if session_id else ""

csv_filename = f"{subject_id}{session_suffix}{space_suffix}_desc-deepFCD-clusters_pthr-{p_thr}_cthr-{c_thr}_results.csv"
nifti_filename = f"{subject_id}{session_suffix}{space_suffix}_desc-deepFCD-clusters_pthr-{p_thr}_cthr-{c_thr}_mask.nii.gz"

csv_path = output_dir / csv_filename
nifti_path = output_dir / nifti_filename

# Save CSV results
ranked_results[cols_pref].to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")

# Save clustered image
nib.save(out_scan, nifti_path)
print(f"Cluster mask saved to: {nifti_path}")
