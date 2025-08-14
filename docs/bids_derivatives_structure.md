# deepFCD BIDS Derivatives Structure

This document describes the output structure created by the deepFCD pipeline when processing BIDS datasets.

## Output Directory Structure

When running deepFCD on a BIDS dataset, the output follows the BIDS derivatives specification with two separate derivative datasets:

### 1. Preprocessing Derivatives (`deepFCD-preproc`)
Contains preprocessed images (brain extraction, registration, bias correction):

```
<bids_root>/
├── derivatives/
│   ├── deepFCD-preproc/
│   │   ├── dataset_description.json
│   │   ├── sub-<subject>/
│   │   │   ├── [ses-<session>/]
│   │   │   │   └── anat/
│   │   │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152NLin2009aSym_label-brain_T1w.nii.gz
│   │   │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152NLin2009aSym_label-brain_FLAIR.nii.gz
│   │   │   │       ├── sub-<subject>[_ses-<session>]_from-T1w_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   │   │   │       └── sub-<subject>[_ses-<session>]_from-FLAIR_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   │   │   └── anat/  # (if no sessions)
│   │   └── sub-<subject2>/
│   │       └── ...
```

### 2. Inference Derivatives (`deepFCD`)
Contains FCD detection results:

```
<bids_root>/
├── derivatives/
│   └── deepFCD/
│       ├── dataset_description.json
│       ├── sub-<subject>/
│       │   ├── [ses-<session>/]
│       │   │   └── func/
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
│       │   │       └── sub-<subject>[_ses-<session>]_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
│       │   └── func/  # (if no sessions)
│       │       ├── sub-<subject>_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
│       │       └── sub-<subject>_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
│       └── sub-<subject2>/
│           └── ...
```

## File Naming Convention

### Preprocessing Files (`deepFCD-preproc`)
- **Preprocessed images**: `sub-<subject>[_ses-<session>]_space-<space>_label-brain_<suffix>.nii.gz`
- **Transform files**: `sub-<subject>[_ses-<session>]_from-<source>_to-<target>_mode-image_xfm.mat`

### Inference Files (`deepFCD`)
The output files follow BIDS derivatives naming conventions:

- **Base structure**: `sub-<subject>[_ses-<session>]_space-<space>_desc-<description>_<suffix>.nii.gz`
- **Subject**: Always prefixed with `sub-`
- **Session**: Included when present in the dataset, prefixed with `ses-`
- **Space**: Always `MNI152NLin2009aSym` (standard MNI space)
- **Description**: `deepFCD` to identify the processing pipeline
- **Suffix**: 
  - `probseg-mean`: Mean probability segmentation
  - `probseg-var`: Variance of probability segmentation

## Examples

### Dataset without sessions:
```
derivatives/
├── deepFCD-preproc/
│   ├── dataset_description.json
│   ├── sub-001/
│   │   └── anat/
│   │       ├── sub-001_space-MNI152NLin2009aSym_label-brain_T1w.nii.gz
│   │       ├── sub-001_space-MNI152NLin2009aSym_label-brain_FLAIR.nii.gz
│   │       ├── sub-001_from-T1w_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   │       └── sub-001_from-FLAIR_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   └── sub-002/
│       └── anat/
│           ├── sub-002_space-MNI152NLin2009aSym_label-brain_T1w.nii.gz
│           ├── sub-002_space-MNI152NLin2009aSym_label-brain_FLAIR.nii.gz
│           ├── sub-002_from-T1w_to-MNI152NLin2009aSym_mode-image_xfm.mat
│           └── sub-002_from-FLAIR_to-MNI152NLin2009aSym_mode-image_xfm.mat
└── deepFCD/
    ├── dataset_description.json
    ├── sub-001/
    │   └── func/
    │       ├── sub-001_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
    │       └── sub-001_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
    └── sub-002/
        └── func/
            ├── sub-002_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
            └── sub-002_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
```

### Dataset with sessions:
```
derivatives/
├── deepFCD-preproc/
│   ├── dataset_description.json
│   ├── sub-001/
│   │   ├── ses-baseline/
│   │   │   └── anat/
│   │   │       ├── sub-001_ses-baseline_space-MNI152NLin2009aSym_label-brain_T1w.nii.gz
│   │   │       ├── sub-001_ses-baseline_space-MNI152NLin2009aSym_label-brain_FLAIR.nii.gz
│   │   │       ├── sub-001_ses-baseline_from-T1w_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   │   │       └── sub-001_ses-baseline_from-FLAIR_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   │   └── ses-followup/
│   │       └── anat/
│   │           ├── sub-001_ses-followup_space-MNI152NLin2009aSym_label-brain_T1w.nii.gz
│   │           ├── sub-001_ses-followup_space-MNI152NLin2009aSym_label-brain_FLAIR.nii.gz
│   │           ├── sub-001_ses-followup_from-T1w_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   │           └── sub-001_ses-followup_from-FLAIR_to-MNI152NLin2009aSym_mode-image_xfm.mat
│   └── sub-002/
│       └── ses-baseline/
│           └── anat/
│               ├── sub-002_ses-baseline_space-MNI152NLin2009aSym_label-brain_T1w.nii.gz
│               ├── sub-002_ses-baseline_space-MNI152NLin2009aSym_label-brain_FLAIR.nii.gz
│               ├── sub-002_ses-baseline_from-T1w_to-MNI152NLin2009aSym_mode-image_xfm.mat
│               └── sub-002_ses-baseline_from-FLAIR_to-MNI152NLin2009aSym_mode-image_xfm.mat
└── deepFCD/
    ├── dataset_description.json
    ├── sub-001/
    │   ├── ses-baseline/
    │   │   └── func/
    │   │       ├── sub-001_ses-baseline_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
    │   │       └── sub-001_ses-baseline_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
    │   └── ses-followup/
    │       └── func/
    │           ├── sub-001_ses-followup_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
    │           └── sub-001_ses-followup_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
    └── sub-002/
        └── ses-baseline/
            └── func/
                ├── sub-002_ses-baseline_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
                └── sub-002_ses-baseline_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
```

## File Contents

### Preprocessing Files (`deepFCD-preproc`)
- **T1w/FLAIR brain-extracted**: Skull-stripped images in MNI152 space
- **Transform files**: Transformation matrices for mapping back to original space

### Inference Files (`deepFCD`)
- **probseg-mean.nii.gz**: Mean probability map for FCD detection across multiple model runs
- **probseg-var.nii.gz**: Variance/uncertainty map showing model confidence

## Usage

To generate outputs in this structure, simply run:

```bash
python inference_bids.py -bp /path/to/bids/dataset -bm -pp
```

The derivatives will be automatically created in:
- `<bids_dataset>/derivatives/deepFCD-preproc/` (preprocessing outputs)
- `<bids_dataset>/derivatives/deepFCD/` (inference outputs)

To specify a custom output location for inference results:

```bash
python inference_bids.py -bp /path/to/bids/dataset -op /custom/output/path -bm -pp
```

Note: Preprocessing outputs will always be placed in `<bids_dataset>/derivatives/deepFCD-preproc/` regardless of the `-op` option.

### Output Checking and Overwrite Behavior

The pipeline automatically checks for existing outputs and skips processing when outputs already exist:

**Preprocessing**: If preprocessed files (brain-extracted images) already exist for a subject/session, preprocessing will be skipped.

**Inference**: If prediction files already exist for a subject/session, inference will be skipped.

To force reprocessing of existing outputs, use the appropriate overwrite flags:

```bash
# Force reprocessing of inference outputs only
python inference_bids.py -bp /path/to/bids/dataset -bm -pp --overwrite

# Force reprocessing of preprocessing outputs only  
python inference_bids.py -bp /path/to/bids/dataset -bm -pp --overwrite-preprocessing

# Force reprocessing of both preprocessing and inference outputs
python inference_bids.py -bp /path/to/bids/dataset -bm -pp --overwrite --overwrite-preprocessing
```

This separation is useful because:
- **Preprocessing** is computationally expensive and outputs rarely need to be regenerated
- **Inference** outputs might need to be regenerated more frequently (e.g., with updated models)
- You can selectively regenerate only what you need

The pipeline will log which subjects are being processed vs. skipped for transparency.
