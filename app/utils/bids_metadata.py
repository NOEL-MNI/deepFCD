#!/usr/bin/env python3
"""
BIDS metadata generation utilities for deepFCD pipeline.
"""

import json
import os
import datetime
from typing import Dict, List, Optional, Tuple


def get_image_metadata(image_path: str) -> Dict:
    """Extract basic metadata from image file path.
    
    Args:
        image_path: Path to the NIfTI image file
        
    Returns:
        Dictionary containing basic image metadata
    """
    try:
        # For now, return basic metadata that doesn't require nibabel
        # This can be enhanced later when nibabel is available
        metadata = {
            "ImageType": ["DERIVED", "PRIMARY"],
            "ProcessingNote": "Metadata extracted from file path only. Install nibabel for detailed image properties.",
            "Extension": os.path.splitext(image_path)[1]
        }
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not extract metadata from {image_path}: {e}")
        return {}


def generate_preprocessing_metadata(
    subject_id: str,
    session_id: Optional[str],
    modality: str,
    output_path: str,
    original_files: List[str],
    processing_steps: List[str],
    software_versions: Dict[str, str] = None,
    space: str = "MNI152NLin2009aSym"
) -> Dict:
    """Generate BIDS-compliant metadata for preprocessed files.
    
    Args:
        subject_id: Subject identifier (e.g., 'sub-001')
        session_id: Session identifier (e.g., 'ses-01') or None
        modality: Image modality ('T1w' or 'FLAIR')  
        output_path: Path to the output preprocessed file
        original_files: List of original input files used
        processing_steps: List of processing steps applied
        software_versions: Dictionary of software versions used
        space: Coordinate space of the output image
        
    Returns:
        Dictionary containing BIDS metadata
    """
    if software_versions is None:
        software_versions = {
            "deepMask": "latest",
            "ANTs": "2.3.3+",
            "N3BiasFieldCorrection": "ITK-based"
        }
    
    # Get current timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    # Extract image-specific metadata if file exists
    image_metadata = {}
    if os.path.exists(output_path):
        image_metadata = get_image_metadata(output_path)
    
    # Generate base metadata
    metadata = {
        "Description": f"Brain-extracted and preprocessed {modality} anatomical image",
        "Units": "arbitrary",
        "SpatialReference": space,
        "SkullStripped": True,
        "ProcessingSteps": processing_steps,
        "SoftwareVersions": software_versions,
        "IntendedFor": "deepFCD focal cortical dysplasia detection",
        "Sources": original_files,
        "GeneratedBy": [
            {
                "Name": "deepFCD-preproc",
                "Version": "1.0.0",
                "Description": "Preprocessing pipeline for deepFCD",
                "CodeURL": "https://github.com/NOEL-MNI/deepFCD"
            }
        ],
        "ProcessingTimestamp": timestamp
    }
    
    # Add image-specific metadata
    metadata.update(image_metadata)
    
    # Add modality-specific information
    if modality == "T1w":
        metadata["SequenceType"] = "T1-weighted"
        metadata["ContrastType"] = "T1w"
    elif modality == "FLAIR":
        metadata["SequenceType"] = "T2-FLAIR" 
        metadata["ContrastType"] = "FLAIR"
        metadata["InversionTime"] = None  # Would need to be extracted from DICOM
    
    return metadata


def write_json_sidecar(metadata: Dict, json_path: str) -> None:
    """Write metadata to a JSON sidecar file.
    
    Args:
        metadata: Dictionary containing metadata
        json_path: Path where JSON file should be written
    """
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"Generated BIDS metadata: {json_path}")
        
    except Exception as e:
        print(f"Error writing JSON sidecar {json_path}: {e}")


def generate_dataset_description(
    output_dir: str,
    pipeline_name: str = "deepFCD-preproc",
    description: str = "Preprocessing pipeline for deepFCD: brain extraction, registration, and bias correction"
) -> None:
    """Generate or update dataset_description.json for derivatives.
    
    Args:
        output_dir: Root directory of the derivatives dataset
        pipeline_name: Name of the pipeline
        description: Description of the pipeline
    """
    dataset_desc_path = os.path.join(output_dir, "dataset_description.json")
    
    dataset_description = {
        "Name": pipeline_name,
        "BIDSVersion": "1.7.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": pipeline_name,
                "Version": "1.0.0",
                "Description": description,
                "CodeURL": "https://github.com/NOEL-MNI/deepFCD"
            }
        ],
        "HowToAcknowledge": "Please cite the deepFCD paper when using this software.",
        "PipelineDescription": {
            "Name": pipeline_name,
            "Version": "1.0.0",
            "Description": description
        },
        "SourceDatasets": [
            {
                "Description": "Raw BIDS dataset containing T1w and FLAIR anatomical images"
            }
        ],
        "Sources": [
            {
                "Suffix": "T1w",
                "Description": "T1-weighted anatomical images",
                "Units": "arbitrary"
            },
            {
                "Suffix": "FLAIR", 
                "Description": "T2-FLAIR anatomical images",
                "Units": "arbitrary"
            }
        ]
    }
    
    # Write dataset description if it doesn't exist
    if not os.path.exists(dataset_desc_path):
        write_json_sidecar(dataset_description, dataset_desc_path)


def generate_bids_metadata_for_outputs(
    subject_id: str,
    session_id: Optional[str],
    output_dir: str,
    original_t1_file: str,
    original_t2_file: str,
    processing_steps: List[str],
    space: str = "MNI152NLin2009aSym"
) -> None:
    """Generate BIDS metadata for both T1 and T2 final output files.
    
    Args:
        subject_id: Subject identifier (e.g., 'sub-001')
        session_id: Session identifier (e.g., 'ses-01') or None
        output_dir: Directory containing the output files
        original_t1_file: Path to original T1w file
        original_t2_file: Path to original T2/FLAIR file
        processing_steps: List of processing steps applied
        space: Coordinate space of output images
    """
    # Construct the full subject/session ID
    if session_id:
        fullid = f"{subject_id}_{session_id}"
    else:
        fullid = subject_id
    
    # Define output file paths
    t1_output = os.path.join(output_dir, f"{fullid}_space-{space}_T1w_brain.nii.gz")
    t2_output = os.path.join(output_dir, f"{fullid}_space-{space}_FLAIR_brain.nii.gz")
    
    # Define JSON sidecar paths
    t1_json = os.path.join(output_dir, f"{fullid}_space-{space}_T1w_brain.json")
    t2_json = os.path.join(output_dir, f"{fullid}_space-{space}_FLAIR_brain.json")
    
    # Software versions
    software_versions = {
        "deepMask": "latest",
        "ANTs": "2.3.3+",
        "N3BiasFieldCorrection": "ITK-based",
        "Python": "3.8+",
        "ANTsPy": "0.4.2+"
    }
    
    # Generate T1w metadata
    if os.path.exists(t1_output):
        t1_metadata = generate_preprocessing_metadata(
            subject_id=subject_id,
            session_id=session_id,
            modality="T1w",
            output_path=t1_output,
            original_files=[original_t1_file],
            processing_steps=processing_steps,
            software_versions=software_versions,
            space=space
        )
        write_json_sidecar(t1_metadata, t1_json)
    
    # Generate T2/FLAIR metadata  
    if os.path.exists(t2_output):
        t2_metadata = generate_preprocessing_metadata(
            subject_id=subject_id,
            session_id=session_id,
            modality="FLAIR",
            output_path=t2_output,
            original_files=[original_t2_file],
            processing_steps=processing_steps,
            software_versions=software_versions,
            space=space
        )
        write_json_sidecar(t2_metadata, t2_json)
    
    # Ensure dataset description exists
    # Go up the directory tree to find the derivatives root
    derivatives_root = output_dir
    while not os.path.basename(derivatives_root).startswith('deepFCD'):
        parent = os.path.dirname(derivatives_root)
        if parent == derivatives_root:  # Reached filesystem root
            break
        derivatives_root = parent
    
    generate_dataset_description(derivatives_root)


def parse_subject_session(fullid: str) -> Tuple[str, Optional[str]]:
    """Parse full subject ID into subject and session components.
    
    Args:
        fullid: Full ID like 'sub-001' or 'sub-001_ses-01'
        
    Returns:
        Tuple of (subject_id, session_id)
    """
    if '_ses-' in fullid:
        subject_id, session_part = fullid.split('_ses-', 1)
        session_id = f"ses-{session_part}"
        return subject_id, session_id
    else:
        return fullid, None


def generate_inference_metadata(
    subject_id: str,
    session_id: Optional[str],
    modality: str,
    output_path: str,
    processing_description: str,
    space: str = "MNI152NLin2009aSym"
) -> Dict:
    """Generate BIDS-compliant metadata for inference output files.
    
    Args:
        subject_id: Subject identifier (e.g., 'sub-001')
        session_id: Session identifier (e.g., 'ses-01') or None
        modality: Type of output ('probseg-mean' or 'probseg-var')
        output_path: Path to the output file
        processing_description: Description of the processing
        space: Coordinate space of the output image
        
    Returns:
        Dictionary containing BIDS metadata
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    # Extract image-specific metadata if file exists
    image_metadata = {}
    if os.path.exists(output_path):
        image_metadata = get_image_metadata(output_path)
    
    # Generate base metadata
    metadata = {
        "Description": processing_description,
        "Units": "probability" if "probseg" in modality else "arbitrary",
        "SpatialReference": space,
        "GeneratedBy": [
            {
                "Name": "deepFCD",
                "Version": "1.0.0",
                "Description": "Deep learning-based Focal Cortical Dysplasia detection",
                "CodeURL": "https://github.com/NOEL-MNI/deepFCD"
            }
        ],
        "ProcessingTimestamp": timestamp,
        "RawSources": []  # Will be populated with preprocessed input files
    }
    
    # Add image-specific metadata
    metadata.update(image_metadata)
    
    # Add modality-specific information
    if "probseg-mean" in modality:
        metadata["StatisticalMap"] = "mean"
        metadata["TaskName"] = "FCD detection"
        metadata["ContrastDefinition"] = "Probability of focal cortical dysplasia presence"
        metadata["IntendedFor"] = "Statistical analysis and visualization of FCD detection results"
    elif "probseg-var" in modality:
        metadata["StatisticalMap"] = "variance"
        metadata["TaskName"] = "FCD detection uncertainty"
        metadata["ContrastDefinition"] = "Uncertainty/variance in FCD detection probability"
        metadata["IntendedFor"] = "Uncertainty quantification for FCD detection results"
    
    return metadata


def generate_inference_bids_metadata(options, uncertainty=True):
    """Generate BIDS metadata for inference output files.
    
    Args:
        options: Dictionary containing inference options including file paths
        uncertainty: Whether uncertainty files were generated
    """
    try:
        if generate_bids_metadata_for_inference_outputs is not None and parse_subject_session is not None:
            # Extract subject information from fullid
            fullid = options.get("fullid", "")
            if not fullid:
                print("Warning: fullid not found in options, skipping BIDS metadata generation")
                return
                
            subject_id, session_id = parse_subject_session(fullid)
            
            # Get file paths
            mean_file = options.get("test_mean_name", "")
            var_file = options.get("test_var_name", "") if uncertainty else ""
            
            if not mean_file:
                print("Warning: test_mean_name not found in options, skipping BIDS metadata generation")
                return
                
            # Get output directory from file path
            output_dir = os.path.dirname(mean_file)
            
            # Determine preprocessed sources (these would be passed in the options ideally)
            preprocessed_sources = []
            if "orig_files" in options:
                preprocessed_sources = options["orig_files"]
            
            # Generate metadata
            generate_bids_metadata_for_inference_outputs(
                subject_id=subject_id,
                session_id=session_id,
                output_dir=output_dir,
                mean_file_path=mean_file,
                var_file_path=var_file,
                preprocessed_sources=preprocessed_sources,
                space="MNI152NLin2009aSym"  # Assuming MNI space for inference outputs
            )
            
            print(f"Generated BIDS metadata for inference outputs: {fullid}")
            
        else:
            print("Warning: BIDS metadata utilities not available for inference - skipping metadata generation")
            
    except Exception as e:
        print(f"Warning: Error generating BIDS metadata for inference: {e}")


def generate_bids_metadata_for_inference_outputs(
    subject_id: str,
    session_id: Optional[str],
    output_dir: str,
    mean_file_path: str,
    var_file_path: str,
    preprocessed_sources: List[str],
    space: str = "MNI152NLin2009aSym"
) -> None:
    """Generate BIDS metadata for inference output files.
    
    Args:
        subject_id: Subject identifier (e.g., 'sub-001')
        session_id: Session identifier (e.g., 'ses-01') or None
        output_dir: Directory containing the output files
        mean_file_path: Path to the mean probability segmentation file
        var_file_path: Path to the variance probability segmentation file
        preprocessed_sources: List of preprocessed input files used
        space: Coordinate space of output images
    """
    # Generate metadata for mean probability segmentation
    if os.path.exists(mean_file_path):
        mean_metadata = generate_inference_metadata(
            subject_id=subject_id,
            session_id=session_id,
            modality="probseg-mean",
            output_path=mean_file_path,
            processing_description="Mean probability map for focal cortical dysplasia detection using deep learning",
            space=space
        )
        mean_metadata["RawSources"] = preprocessed_sources
        
        mean_json_path = mean_file_path.replace('.nii.gz', '.json').replace('.nii', '.json')
        write_json_sidecar(mean_metadata, mean_json_path)
    
    # Generate metadata for variance probability segmentation
    if os.path.exists(var_file_path):
        var_metadata = generate_inference_metadata(
            subject_id=subject_id,
            session_id=session_id,
            modality="probseg-var",
            output_path=var_file_path,
            processing_description="Variance/uncertainty map for focal cortical dysplasia detection using deep learning",
            space=space
        )
        var_metadata["RawSources"] = preprocessed_sources
        
        var_json_path = var_file_path.replace('.nii.gz', '.json').replace('.nii', '.json')
        write_json_sidecar(var_metadata, var_json_path)
    
    # Ensure dataset description exists in the derivatives root
    derivatives_root = output_dir
    while not os.path.basename(derivatives_root).startswith('deepFCD'):
        parent = os.path.dirname(derivatives_root)
        if parent == derivatives_root:  # Reached filesystem root
            break
        derivatives_root = parent
    
    generate_dataset_description(
        derivatives_root, 
        pipeline_name="deepFCD",
        description="Deep learning-based Focal Cortical Dysplasia detection"
    )
