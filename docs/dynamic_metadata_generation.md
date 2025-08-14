# Dynamic BIDS Metadata Generation for deepFCD

This implementation provides automatic generation of BIDS-compliant metadata as preprocessed outputs and inference results are created by the deepFCD pipeline.

## Features Implemented

### 📁 **File Structure**
```
deepFCD/
├── app/
│   └── utils/
│       └── bids_metadata.py          # Core metadata generation utilities
├── bids_inputs.json                   # Input specification
├── bids_validation_config.json       # Validation configuration  
├── templates/                         # Template JSON files
│   ├── dataset_description_preproc.json
│   ├── dataset_description_inference.json
│   ├── t1_brain_final.json
│   └── t2_brain_final.json
├── docs/
│   └── input_requirements.md         # Comprehensive input documentation
└── test_bids_metadata.py            # Test script for validation
```

### 🔄 **Dynamic Generation Points**

#### 1. Preprocessing Stage
**Location**: `app/deepMask/app/utils/image_processing.py`
- **Trigger**: After `ants.image_write()` calls in `__skull_stripping()` method
- **Generated Files**:
  - `*_t1_brain_final.json` - T1w preprocessing metadata
  - `*_t2_brain_final.json` - FLAIR/T2 preprocessing metadata
  - `dataset_description.json` - Preprocessing derivatives description

#### 2. Inference Stage  
**Location**: `app/utils/base.py`
- **Trigger**: After `out_scan.to_filename()` calls in `test_model()` function
- **Generated Files**:
  - `*_probseg-mean.json` - Mean probability segmentation metadata
  - `*_probseg-var.json` - Variance/uncertainty metadata
  - `dataset_description.json` - Inference derivatives description

### 📋 **Generated Metadata Content**

#### Preprocessing Metadata
```json
{
  "Description": "Brain-extracted and preprocessed T1w anatomical image",
  "Units": "arbitrary",
  "SpatialReference": "MNI152NLin2009aSym",
  "SkullStripped": true,
  "ProcessingSteps": [
    "Brain extraction using deepMask neural network",
    "N3 bias field correction",
    "Intensity normalization",
    "Registration to MNI152NLin2009aSym template space using ANTs"
  ],
  "SoftwareVersions": {
    "deepMask": "latest",
    "ANTs": "2.3.3+",
    "N3BiasFieldCorrection": "ITK-based"
  },
  "Sources": ["/path/to/original/T1w.nii.gz"],
  "ProcessingTimestamp": "2025-08-09T10:30:45.123456"
}
```

#### Inference Metadata
```json
{
  "Description": "Mean probability map for focal cortical dysplasia detection using deep learning",
  "Units": "probability",
  "SpatialReference": "MNI152NLin2009aSym",
  "StatisticalMap": "mean",
  "TaskName": "FCD detection",
  "ContrastDefinition": "Probability of focal cortical dysplasia presence",
  "RawSources": [
    "/path/to/sub-001_t1_brain_final.nii.gz",
    "/path/to/sub-001_t2_brain_final.nii.gz"
  ],
  "ProcessingTimestamp": "2025-08-09T10:35:22.789012"
}
```

### 🛠 **Implementation Details**

#### Core Functions

1. **`generate_bids_metadata_for_outputs()`**
   - Generates metadata for preprocessing outputs
   - Called automatically after brain extraction
   - Creates JSON sidecars for T1 and T2 final outputs

2. **`generate_bids_metadata_for_inference_outputs()`**
   - Generates metadata for inference results
   - Called automatically after model prediction
   - Creates JSON sidecars for probability maps

3. **`parse_subject_session()`**
   - Extracts subject and session IDs from BIDS identifiers
   - Handles both session and non-session datasets

4. **`generate_dataset_description()`**
   - Creates BIDS derivatives dataset descriptions
   - Ensures proper provenance tracking

#### Integration Points

1. **Preprocessing Integration**:
   ```python
   # In image_processing.py __skull_stripping() method
   ants.image_write(self._t1_n4 * self._mask, self._t1brainfile)
   ants.image_write(self._t2_n4 * self._mask, self._t2brainfile)
   
   # Generate BIDS metadata for the output files
   self.__generate_bids_metadata()
   ```

2. **Inference Integration**:
   ```python
   # In base.py test_model() function
   out_scan.to_filename(options["test_mean_name"])
   if uncertainty:
       out_scan.to_filename(options["test_var_name"])
       
   # Generate BIDS metadata for inference outputs
   generate_inference_bids_metadata(options, uncertainty)
   ```

### 🧪 **Testing**

Run the test script to validate metadata generation:

```bash
python test_bids_metadata.py
```

**Expected Output**:
```
Testing BIDS metadata generation for deepFCD...
==================================================
✓ Successfully imported BIDS metadata utilities
✓ Subject/session parsing works correctly
✓ Preprocessing metadata generation works correctly
✓ T1 metadata content is valid
✓ Inference metadata generation works correctly
✓ Mean probseg metadata content is valid
✓ All BIDS metadata generation tests passed!

🎉 All tests passed! BIDS metadata generation is working correctly.
```

### 📝 **Usage**

The metadata generation is **completely automatic** - no additional parameters or commands are needed. When you run:

```bash
python app/inference_bids.py -bp <bids_directory> -sp MNI152NLin2009aSym -pp -bm
```

The pipeline will now:

1. **During preprocessing**: Generate JSON sidecars for `*_t1_brain_final.nii.gz` and `*_t2_brain_final.nii.gz`
2. **During inference**: Generate JSON sidecars for `*_probseg-mean.nii.gz` and `*_probseg-var.nii.gz`
3. **Create dataset descriptions**: Ensure BIDS derivatives have proper `dataset_description.json` files

### 🔍 **BIDS Compliance Features**

- **Provenance Tracking**: All files include source file references
- **Processing Documentation**: Detailed processing steps recorded
- **Software Versions**: Version information preserved
- **Timestamps**: Processing time recorded
- **BIDS Entities**: Proper use of BIDS entities (space, desc, suffix)
- **Units and Definitions**: Clear definitions of data units and contrasts

### 🚨 **Error Handling**

The implementation includes robust error handling:
- Graceful fallback if metadata utilities are unavailable
- Warning messages for missing information
- Continues processing even if metadata generation fails
- Validation of required fields before generation

### 📈 **Benefits**

1. **BIDS Compliance**: Outputs fully comply with BIDS derivatives specification
2. **Reproducibility**: Complete provenance chain from raw data to results
3. **Automatic**: No manual intervention required
4. **Extensible**: Easy to add new metadata fields or processing steps
5. **Robust**: Handles various dataset structures and edge cases

This implementation ensures that deepFCD outputs are fully BIDS-compliant and include rich metadata for reproducibility and interoperability with other neuroimaging tools.
