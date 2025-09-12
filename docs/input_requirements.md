# deepFCD BIDS Input Requirements

This document describes the input requirements for the deepFCD focal cortical dysplasia detection pipeline.

## Required Input Files

The deepFCD pipeline requires the following preprocessed anatomical images:

### 1. T1-weighted Image
- **Filename pattern**: `*_t1_brain_final.nii.gz`
- **Original BIDS suffix**: `T1w`
- **Description**: Brain-extracted, bias-corrected T1-weighted anatomical image
- **Preprocessing requirements**:
  - Brain extraction (skull stripping)
  - N3/N4 bias field correction
  - Intensity normalization
  - Registration to MNI152 space (recommended)

### 2. T2-FLAIR Image  
- **Filename pattern**: `*_t2_brain_final.nii.gz`
- **Original BIDS suffix**: `FLAIR`
- **Description**: Brain-extracted, bias-corrected T2-FLAIR anatomical image
- **Preprocessing requirements**:
  - Brain extraction (skull stripping)
  - N3/N4 bias field correction  
  - Intensity normalization
  - Registration to MNI152 space (recommended)

## Input Specifications

### File Format
- **Extension**: `.nii.gz` (compressed NIfTI)
- **Orientation**: RAS or LAS
- **Resolution**: 1×1×1 mm (preferred)
- **Coordinate Space**: MNI152NLin2009aSym (recommended)

### Image Properties
- **Intensity Range**: Positive values only (after bias correction and normalization)
- **Brain Extraction**: Required (skull-stripped images)
- **Bias Correction**: Required (N3/N4 bias field correction)

## Automatic Preprocessing

If raw T1w and FLAIR images are provided, deepFCD can automatically perform preprocessing:

```bash
python app/inference_bids.py -bp <bids_directory> -sp MNI152NLin2009aSym -pp -bm
```

The `-pp` flag enables preprocessing, which will:
1. Apply brain extraction using deepMask
2. Perform N3 bias field correction
3. Normalize intensity values
4. Register images to MNI152 space
5. Generate the required `*_t1_brain_final.nii.gz` and `*_t2_brain_final.nii.gz` files

## BIDS Structure

### Input Dataset Structure
```
bids_root/
├── sub-001/
│   ├── anat/
│   │   ├── sub-001_T1w.nii.gz
│   │   ├── sub-001_T1w.json
│   │   ├── sub-001_FLAIR.nii.gz
│   │   └── sub-001_FLAIR.json
│   └── ...
└── ...
```

### Output Derivatives Structure
```
bids_root/
├── derivatives/
│   ├── deepFCD-preproc/
│   │   ├── dataset_description.json
│   │   └── sub-001/
│   │       └── preproc/
│   │           ├── sub-001_t1_brain_final.nii.gz
│   │           ├── sub-001_t1_brain_final.json
│   │           ├── sub-001_t2_brain_final.nii.gz
│   │           └── sub-001_t2_brain_final.json
│   └── deepFCD/
│       ├── dataset_description.json
│       └── sub-001/
│           └── func/
│               ├── sub-001_space-MNI152NLin2009aSym_desc-deepFCD_probseg-mean.nii.gz
│               └── sub-001_space-MNI152NLin2009aSym_desc-deepFCD_probseg-var.nii.gz
└── ...
```

## Quality Control

Before running deepFCD inference, ensure that:

1. **Both T1w and FLAIR images are available** for each subject/session
2. **Images are properly brain-extracted** (no skull or non-brain tissue)
3. **Bias field correction has been applied** (uniform intensity across the brain)
4. **Images are in the same coordinate space** (preferably MNI152)
5. **Image resolution is adequate** (1×1×1 mm recommended)
6. **Intensity values are positive** and properly normalized

## Example Usage

### With preprocessing enabled:
```bash
python app/inference_bids.py \
    -bp /path/to/bids/dataset \
    -sp MNI152NLin2009aSym \
    -dev cuda \
    -pp \
    -bm
```

### With pre-existing preprocessed files:
```bash
python app/inference_bids.py \
    -bp /path/to/bids/dataset \
    -sp MNI152NLin2009aSym \
    -dev cuda
```

## Troubleshooting

### "No inputs found for deepFCD"
This error occurs when the required `*_t1_brain_final.nii.gz` and `*_t2_brain_final.nii.gz` files are not found. Solutions:

1. **Enable preprocessing** with the `-pp` flag if you have raw T1w and FLAIR images
2. **Check file naming** - ensure preprocessed files follow the exact naming convention
3. **Verify file locations** - files should be in the expected BIDS derivatives structure
4. **Check both modalities** - both T1 and FLAIR preprocessed files must be present

### Missing Modalities
- Ensure both T1w and FLAIR images are available in your BIDS dataset
- Check that the BIDS naming convention is followed correctly
- Verify that both modalities are acquired for each subject/session

## References

For more information about BIDS derivatives specification:
- [BIDS Specification](https://bids-specification.readthedocs.io/)
- [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/05-derivatives/01-introduction.html)
