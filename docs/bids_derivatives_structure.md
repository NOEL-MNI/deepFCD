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
│   │   │   │   ├── anat/
│   │   │   │   │   ├── sub-<subject>[_ses-<session>]_space-MNI152_label-brain_T1w.nii.gz
│   │   │   │   │   └── sub-<subject>[_ses-<session>]_space-MNI152_label-brain_FLAIR.nii.gz
│   │   │   │   └── xfm/
│   │   │   │       ├── sub-<subject>[_ses-<session>]_from-T1w_to-MNI152_mode-image_xfm.mat
│   │   │   │       └── sub-<subject>[_ses-<session>]_from-FLAIR_to-MNI152_mode-image_xfm.mat
│   │   │   └── anat/  # (if no sessions)
│   │   │       ├── sub-<subject>_space-MNI152_label-brain_T1w.nii.gz
│   │   │       └── sub-<subject>_space-MNI152_label-brain_FLAIR.nii.gz
│   │   │   └── xfm/   # (if no sessions)
│   │   │       ├── sub-<subject>_from-T1w_to-MNI152_mode-image_xfm.mat
│   │   │       └── sub-<subject>_from-FLAIR_to-MNI152_mode-image_xfm.mat
│   │   └── sub-<subject2>/
│   │       └── ...
```

### 2. Inference Derivatives (`deepFCD`)
Contains FCD detection results with dual CNN outputs and optional postprocessing:

```
<bids_root>/
├── derivatives/
│   └── deepFCD/
│       ├── dataset_description.json
│       ├── sub-<subject>/
│       │   ├── [ses-<session>/]
│       │   │   └── anat/
│       │   │       ├── # MNI152 space outputs (always generated)
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-mean0_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-var0_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-mean1_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-var1_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD-postproc_mask.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD-postproc_label.nii.gz
│       │   │       ├── # Original space outputs (transformed back)
│       │   │       ├── sub-<subject>[_ses-<session>]_space-orig_desc-deepFCD_stat-mean0_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-orig_desc-deepFCD_stat-var0_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-orig_desc-deepFCD_stat-mean1_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_space-orig_desc-deepFCD_stat-var1_probseg.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_desc-deepFCD-postproc_mask.nii.gz
│       │   │       ├── sub-<subject>[_ses-<session>]_desc-deepFCD-postproc_label.nii.gz
│       │   │       ├── # BIDS metadata files
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-mean0_probseg.json
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-var0_probseg.json
│       │   │       ├── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-mean1_probseg.json
│       │   │       └── sub-<subject>[_ses-<session>]_space-MNI152_desc-deepFCD_stat-var1_probseg.json
│       │   └── anat/  # (if no sessions)
│       │       └── # Similar structure without session identifier
│       └── sub-<subject2>/
│           └── ...
```

## File Naming Convention

### Preprocessing Files (`deepFCD-preproc`)
- **Preprocessed images**: `sub-<subject>[_ses-<session>]_space-<space>_label-brain_<suffix>.nii.gz`
- **Transform files**: `sub-<subject>[_ses-<session>]_from-<source>_to-<target>_mode-image_xfm.mat`

**Note**: Transform files follow BIDS Enhancement Proposal 14 (BEP014) for spaces and mappings, using the `_xfm.mat` suffix and stored in dedicated `xfm/` directories at the session/subject level.

### Inference Files (`deepFCD`)
The output files follow BIDS derivatives naming conventions with custom entities:

- **Base structure**: `sub-<subject>[_ses-<session>]_space-<space>_desc-<description>_stat-<statistic>_<suffix>.nii.gz`
- **Subject**: Always prefixed with `sub-`
- **Session**: Included when present in the dataset, prefixed with `ses-`
- **Space**: 
  - `MNI152`: Standard MNI space (always generated)
  - `orig`: Original/native space (transformed back from MNI152)
- **Description**: 
  - `deepFCD`: Main inference outputs
  - `deepFCD-postproc`: Post-processed segmentation outputs
- **Statistics** (custom entity):
  - `mean0`: Mean probability from first CNN
  - `var0`: Variance/uncertainty from first CNN
  - `mean1`: Mean probability from second CNN (final output)
  - `var1`: Variance/uncertainty from second CNN (final output)
- **Suffix**: 
  - `probseg`: Probability segmentation maps
  - `mask`: Binary masks (post-processed)
  - `label`: Labeled regions (post-processed)

## Examples

### Dataset without sessions:
```
derivatives/
├── deepFCD-preproc/
│   ├── dataset_description.json
│   ├── sub-001/
│   │   ├── anat/
│   │   │   ├── sub-001_space-MNI152_label-brain_T1w.nii.gz
│   │   │   └── sub-001_space-MNI152_label-brain_FLAIR.nii.gz
│   │   └── xfm/
│   │       ├── sub-001_from-T1w_to-MNI152_mode-image_xfm.mat
│   │       └── sub-001_from-FLAIR_to-MNI152_mode-image_xfm.mat
│   └── sub-002/
│       ├── anat/
│       │   ├── sub-002_space-MNI152_label-brain_T1w.nii.gz
│       │   └── sub-002_space-MNI152_label-brain_FLAIR.nii.gz
│       └── xfm/
│           ├── sub-002_from-T1w_to-MNI152_mode-image_xfm.mat
│           └── sub-002_from-FLAIR_to-MNI152_mode-image_xfm.mat
└── deepFCD/
    ├── dataset_description.json
    ├── sub-001/
    │   └── anat/
    │       ├── sub-001_space-MNI152_desc-deepFCD_stat-mean0_probseg.nii.gz
    │       ├── sub-001_space-MNI152_desc-deepFCD_stat-var0_probseg.nii.gz
    │       ├── sub-001_space-MNI152_desc-deepFCD_stat-mean1_probseg.nii.gz
    │       ├── sub-001_space-MNI152_desc-deepFCD_stat-var1_probseg.nii.gz
    │       ├── sub-001_space-MNI152_desc-deepFCD-postproc_mask.nii.gz
    │       ├── sub-001_space-MNI152_desc-deepFCD-postproc_label.nii.gz
    │       ├── sub-001_space-orig_desc-deepFCD_stat-mean0_probseg.nii.gz
    │       ├── sub-001_space-orig_desc-deepFCD_stat-var0_probseg.nii.gz
    │       ├── sub-001_space-orig_desc-deepFCD_stat-mean1_probseg.nii.gz
    │       ├── sub-001_space-orig_desc-deepFCD_stat-var1_probseg.nii.gz
    │       ├── sub-001_desc-deepFCD-postproc_mask.nii.gz
    │       ├── sub-001_desc-deepFCD-postproc_label.nii.gz
    │       └── sub-001_space-MNI152_desc-deepFCD_stat-*.json
    └── sub-002/
        └── anat/
            └── # Similar structure as sub-001
```

### Dataset with sessions:
```
derivatives/
├── deepFCD-preproc/
│   ├── dataset_description.json
│   ├── sub-001/
│   │   ├── ses-baseline/
│   │   │   └── anat/
│   │   │       ├── sub-001_ses-baseline_space-MNI152_label-brain_T1w.nii.gz
│   │   │       ├── sub-001_ses-baseline_space-MNI152_label-brain_FLAIR.nii.gz
│   │   │       ├── sub-001_ses-baseline_from-T1w_to-MNI152_mode-image_xfm.mat
│   │   │       └── sub-001_ses-baseline_from-FLAIR_to-MNI152_mode-image_xfm.mat
│   │   └── ses-followup/
│   │       └── anat/
│   │           ├── sub-001_ses-followup_space-MNI152_label-brain_T1w.nii.gz
│   │           ├── sub-001_ses-followup_space-MNI152_label-brain_FLAIR.nii.gz
│   │           ├── sub-001_ses-followup_from-T1w_to-MNI152_mode-image_xfm.mat
│   │           └── sub-001_ses-followup_from-FLAIR_to-MNI152_mode-image_xfm.mat
│   └── sub-002/
│       └── ses-baseline/
│           └── anat/
│               ├── sub-002_ses-baseline_space-MNI152_label-brain_T1w.nii.gz
│               ├── sub-002_ses-baseline_space-MNI152_label-brain_FLAIR.nii.gz
│               ├── sub-002_ses-baseline_from-T1w_to-MNI152_mode-image_xfm.mat
│               └── sub-002_ses-baseline_from-FLAIR_to-MNI152_mode-image_xfm.mat
└── deepFCD/
    ├── dataset_description.json
    ├── sub-001/
    │   ├── ses-baseline/
    │   │   └── anat/
    │   │       ├── # MNI152 space outputs
    │   │       ├── sub-001_ses-baseline_space-MNI152_desc-deepFCD_stat-mean0_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_space-MNI152_desc-deepFCD_stat-var0_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_space-MNI152_desc-deepFCD_stat-mean1_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_space-MNI152_desc-deepFCD_stat-var1_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_space-MNI152_desc-deepFCD-postproc_mask.nii.gz
    │   │       ├── sub-001_ses-baseline_space-MNI152_desc-deepFCD-postproc_label.nii.gz
    │   │       ├── # Original space outputs
    │   │       ├── sub-001_ses-baseline_space-orig_desc-deepFCD_stat-mean0_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_space-orig_desc-deepFCD_stat-var0_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_space-orig_desc-deepFCD_stat-mean1_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_space-orig_desc-deepFCD_stat-var1_probseg.nii.gz
    │   │       ├── sub-001_ses-baseline_desc-deepFCD-postproc_mask.nii.gz
    │   │       ├── sub-001_ses-baseline_desc-deepFCD-postproc_label.nii.gz
    │   │       └── # BIDS metadata files (JSON)
    │   └── ses-followup/
    │       └── anat/
    │           └── # Similar structure as ses-baseline
    └── sub-002/
        └── ses-baseline/
            └── anat/
                └── # Similar structure as sub-001/ses-baseline
```

## File Contents

### Preprocessing Files (`deepFCD-preproc`)
- **T1w/FLAIR brain-extracted**: Skull-stripped images in MNI152 space
- **Transform files**: Transformation matrices for mapping back to original space

### Inference Files (`deepFCD`)
- **stat-mean0_probseg.nii.gz**: Mean probability map from first CNN (coarse detection)
- **stat-var0_probseg.nii.gz**: Variance/uncertainty map from first CNN
- **stat-mean1_probseg.nii.gz**: Mean probability map from second CNN (refined detection, final output)
- **stat-var1_probseg.nii.gz**: Variance/uncertainty map from second CNN (final uncertainty)
- **desc-deepFCD-postproc_mask.nii.gz**: Binary mask of detected FCD regions (thresholded)
- **desc-deepFCD-postproc_label.nii.gz**: Labeled regions with cluster IDs
- **JSON metadata files**: BIDS sidecar files containing processing parameters, model information, and provenance

#### Space Variants:
- **space-MNI152**: All outputs in standard MNI152 space (native inference space)
- **space-orig**: Transformed back to original/native subject space for clinical review
- **No space entity**: Post-processed files in original space (mask and label files)

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

## Working with Results

### Cluster Analysis and Reporting

Use the BIDS-compliant reporting script to analyze detected FCD clusters:

```bash
# Basic usage with default thresholds
python utils/reporting_bids.py sub-001 /path/to/bids/derivatives/deepFCD --session ses-baseline

# Custom thresholds and space
python utils/reporting_bids.py sub-001 /path/to/bids/derivatives/deepFCD --session ses-baseline --p_thr 0.8 --c_thr 500 --space orig

# For single-session datasets
python utils/reporting_bids.py sub-001 /path/to/bids/derivatives/deepFCD --space MNI152
```

The reporting script will:
1. Load the final inference outputs (`stat-mean1` and `stat-var1`)
2. Apply probability and cluster size thresholding
3. Rank clusters by confidence and uncertainty
4. Generate anatomical labels using atlas information
5. Save results as CSV tables and cluster masks in BIDS format
