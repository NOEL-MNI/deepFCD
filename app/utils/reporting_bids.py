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
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel import load as load_nii
from sklearn import preprocessing
from tabulate import tabulate
from bids import BIDSLayout
import pandas as pd

from atlasreader.atlasreader import read_atlas_peak
from confidence import extractLesionCluster


def normalize_subject_id(subject_id: str) -> str:
    """Normalize subject ID to BIDS format.
    
    Args:
        subject_id: Subject identifier (with or without 'sub-' prefix)
        
    Returns:
        Normalized subject ID with 'sub-' prefix
    """
    if not subject_id.startswith('sub-'):
        return f'sub-{subject_id}'
    return subject_id


def normalize_session_id(session_id: Optional[str]) -> Optional[str]:
    """Normalize session ID to BIDS format.
    
    Args:
        session_id: Session identifier (with or without 'ses-' prefix), or None
        
    Returns:
        Normalized session ID with 'ses-' prefix, or None if input is None
    """
    if session_id is None:
        return None
    if not session_id.startswith('ses-'):
        return f'ses-{session_id}'
    return session_id


def build_full_identifier(subject_id: str, session_id: Optional[str]) -> str:
    """Build full BIDS identifier from subject and session.
    
    Args:
        subject_id: Normalized subject ID (with 'sub-' prefix)
        session_id: Normalized session ID (with 'ses-' prefix), or None
        
    Returns:
        Full identifier (e.g., 'sub-001' or 'sub-001_ses-01')
    """
    if session_id:
        return f"{subject_id}_{session_id}"
    return subject_id


def find_prediction_files(
    layout: BIDSLayout,
    subject_id: str,
    session_id: Optional[str],
    space: str
) -> Tuple[str, str]:
    """Find mean and variance prediction files using BIDS layout.
    
    Args:
        layout: BIDS layout object for derivatives directory
        subject_id: Normalized subject ID (with 'sub-' prefix)
        session_id: Normalized session ID (with 'ses-' prefix), or None
        space: Space to search for (e.g., 'MNI152', 'orig')
        
    Returns:
        Tuple of (mean_file_path, var_file_path)
        
    Raises:
        FileNotFoundError: If required files are not found
    """
    # Build query parameters
    query_params = {
        'subject': subject_id.replace('sub-', ''),
        'desc': 'deepFCD',
        'suffix': 'probseg',
        'extension': '.nii.gz',
        'space': space
    }
    
    if session_id:
        query_params['session'] = session_id.replace('ses-', '')
    
    # Find all deepFCD files
    all_files = layout.get(**query_params, invalid_filters='allow')
    
    # Filter for mean1 and var1 files manually since stat is not standard BIDS
    mean_files = [f for f in all_files if 'stat-mean1' in f.filename]
    var_files = [f for f in all_files if 'stat-var1' in f.filename]
    
    fullid = build_full_identifier(subject_id, session_id)
    
    if not mean_files:
        error_msg = f"Error: No mean1 prediction files found for {fullid} in space-{space}\n"
        error_msg += f"Search parameters: {query_params}\n"
        error_msg += f"All deepFCD files found: {[f.filename for f in all_files]}\n"
        if all_files:
            error_msg += "Available files by path:\n"
            for f in all_files:
                error_msg += f"  {f.path}\n"
        else:
            error_msg += "No deepFCD files found at all. Check that inference has been run.\n"
        raise FileNotFoundError(error_msg)
    
    if not var_files:
        error_msg = f"Error: No var1 prediction files found for {fullid} in space-{space}\n"
        error_msg += f"Available mean files: {[f.filename for f in mean_files]}\n"
        raise FileNotFoundError(error_msg)
    
    mean_file = mean_files[0].path
    var_file = var_files[0].path
    
    return mean_file, var_file


def setup_options(
    mean_file: str,
    prob_threshold: float,
    cluster_size: int,
    script_dir: Optional[str] = None
) -> Dict:
    """Set up options dictionary for cluster extraction.
    
    Args:
        mean_file: Path to mean prediction file
        prob_threshold: Probability threshold for clustering
        cluster_size: Minimum cluster size threshold
        script_dir: Directory containing the script (for template paths)
        
    Returns:
        Options dictionary for extractLesionCluster
    """
    if script_dir is None:
        script_dir = os.path.realpath(os.path.dirname(__file__))
    
    options = {}
    options["header"] = load_nii(mean_file).header
    options["data_folder"] = str(Path(mean_file).parent)
    options["submask"] = os.path.join(script_dir, "../templates", "subcortical_mask_v3.nii.gz")
    options["t_bin"] = prob_threshold
    options["l_min"] = cluster_size
    
    return options


def extract_clusters(
    fullid: str,
    mean_file: str,
    var_file: str,
    options: Dict
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract lesion clusters from prediction maps.
    
    Args:
        fullid: Full subject identifier
        mean_file: Path to mean prediction file
        var_file: Path to variance prediction file
        options: Options dictionary for cluster extraction
        
    Returns:
        Tuple of (output_scan_array, results_dataframe)
    """
    # Load the data
    ea = load_nii(mean_file).get_fdata()
    ea_var = load_nii(var_file).get_fdata()
    
    # Extract cluster information
    results = {}
    output_scan, results = extractLesionCluster(fullid, ea, ea_var, options)
    
    return output_scan, results


def compute_confidence_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Compute confidence scores from variance values.
    
    Args:
        results: DataFrame with cluster information including 'var' column
        
    Returns:
        DataFrame with added 'confidence' column
    """
    if results.empty:
        return results
    
    results_copy = results.copy()
    results_copy = results_copy.sort_values("rank")
    
    min_max_scaler = preprocessing.MinMaxScaler()
    invert_var = 1 / results_copy["var"]
    results_copy["confidence"] = np.round(
        100.0 * min_max_scaler.fit_transform(invert_var.values.reshape(-1, 1)), 1
    )
    
    return results_copy


def annotate_anatomical_regions(results: pd.DataFrame) -> pd.DataFrame:
    """Annotate clusters with anatomical region labels.
    
    Args:
        results: DataFrame with cluster coordinates
        
    Returns:
        DataFrame with added 'region' column
    """
    if results.empty:
        return results
    
    results_copy = results.copy()
    labels = []
    
    for N in range(len(results_copy.coords)):
        label = read_atlas_peak(
            atlastype="harvard_oxford",
            coordinate=results_copy.coords.iloc[N],
            prob_thresh=5
        )
        labels.append(label[0][1])
    
    results_copy["region"] = labels
    
    return results_copy


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    """Format results for display and export.
    
    Args:
        results: DataFrame with cluster information
        
    Returns:
        Formatted DataFrame with rounded values and cleaned coordinates
    """
    if results.empty:
        return results
    
    results_copy = results.copy()
    results_copy.reset_index(drop=True, inplace=True)
    
    # Round probability and confidence to integers
    results_copy["probability"] = results_copy["probability"].apply(
        lambda x: int(round(x * 100))
    )
    results_copy["confidence"] = results_copy["confidence"].apply(
        lambda x: int(round(x))
    )
    
    # Clean up coordinate formatting
    results_copy["coords"] = results_copy["coords"].apply(
        lambda x: str(x).replace(" ", "").replace("[", "").replace("]", "")
    )
    
    return results_copy


def generate_output_filenames(
    subject_id: str,
    session_id: Optional[str],
    space: str,
    prob_threshold: float,
    cluster_size: int
) -> Tuple[str, str]:
    """Generate BIDS-compliant output filenames.
    
    Args:
        subject_id: Normalized subject ID
        session_id: Normalized session ID or None
        space: Space identifier
        prob_threshold: Probability threshold used
        cluster_size: Cluster size threshold used
        
    Returns:
        Tuple of (csv_filename, nifti_filename)
    """
    space_suffix = f"_space-{space}"
    session_suffix = f"_{session_id}" if session_id else ""
    
    csv_filename = (
        f"{subject_id}{session_suffix}{space_suffix}_"
        f"desc-deepFCD-clusters_pthr-{prob_threshold}_cthr-{cluster_size}_results.csv"
    )
    nifti_filename = (
        f"{subject_id}{session_suffix}{space_suffix}_"
        f"desc-deepFCD-clusters_pthr-{prob_threshold}_cthr-{cluster_size}_mask.nii.gz"
    )
    
    return csv_filename, nifti_filename


def save_results(
    results: pd.DataFrame,
    output_scan: np.ndarray,
    mean_file: str,
    csv_filename: str,
    nifti_filename: str,
    columns: List[str],
    derivatives_dir: str,
    subject_id: str,
    session_id: Optional[str],
    in_place: Optional[str] = None
) -> Tuple[Path, Path]:
    """Save cluster results to CSV and NIfTI files in BIDS derivatives structure.
    
    Args:
        results: Formatted results DataFrame
        output_scan: Clustered output scan array
        mean_file: Path to original mean file (for header/affine)
        csv_filename: Output CSV filename
        nifti_filename: Output NIfTI filename
        columns: Columns to save in CSV
        derivatives_dir: Path to BIDS derivatives directory (parent of deepFCD)
        subject_id: Normalized subject ID
        session_id: Normalized session ID or None
        in_place: If None, save to derivatives/deepFCD-reporting/. 
                  If empty string, save alongside predictions in derivatives/deepFCD/.
                  If a path, save to that directory.
        
    Returns:
        Tuple of (csv_path, nifti_path)
    """
    # Determine output directory based on in_place flag
    if in_place is None:
        # Default: save to derivatives/deepFCD-reporting
        derivatives_parent = Path(derivatives_dir).parent
        reporting_root = derivatives_parent / 'deepFCD-reporting'
        
        # Create dataset_description.json if it doesn't exist
        dataset_desc_path = reporting_root / 'dataset_description.json'
        if not dataset_desc_path.exists():
            reporting_root.mkdir(parents=True, exist_ok=True)
            dataset_description = {
                "Name": "deepFCD Cluster Reporting",
                "BIDSVersion": "1.10.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "deepFCD-reporting",
                        "Version": "2.0.0",
                        "Description": "Cluster-based reporting of deepFCD predictions with anatomical labeling and confidence scores"
                    }
                ],
                "SourceDatasets": [
                    {
                        "URL": "derivatives/deepFCD",
                        "Version": "1.0.0"
                    }
                ]
            }
            with open(dataset_desc_path, 'w') as f:
                json.dump(dataset_description, f, indent=2)
        
        reporting_dir = reporting_root / subject_id
        if session_id:
            reporting_dir = reporting_dir / session_id
        output_dir = reporting_dir / 'anat'
        
    elif in_place == '':
        # In-place: save alongside predictions in derivatives/deepFCD
        reporting_dir = Path(derivatives_dir) / subject_id
        if session_id:
            reporting_dir = reporting_dir / session_id
        output_dir = reporting_dir / 'anat'
        
    else:
        # Custom path provided
        reporting_dir = Path(in_place) / subject_id
        if session_id:
            reporting_dir = reporting_dir / session_id
        output_dir = reporting_dir / 'anat'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / csv_filename
    nifti_path = output_dir / nifti_filename
    
    # Save CSV results
    results[columns].to_csv(csv_path, index=False)
    
    # Save clustered image
    header = load_nii(mean_file).header
    affine = header.get_qform()
    out_scan = nib.Nifti1Image(output_scan, affine=affine, header=header)
    nib.save(out_scan, nifti_path)
    
    return csv_path, nifti_path


def process_subject_clusters(
    subject_id: str,
    derivatives_dir: str,
    session_id: Optional[str] = None,
    space: str = 'MNI152',
    prob_threshold: float = 0.7,
    cluster_size: int = 300,
    in_place: Optional[str] = None
) -> Tuple[pd.DataFrame, Path, Path]:
    """Main processing function to extract and analyze clusters for a subject.
    
    Args:
        subject_id: Subject identifier (with or without 'sub-' prefix)
        derivatives_dir: Path to BIDS derivatives directory
        session_id: Session identifier (with or without 'ses-' prefix), or None
        space: Space to analyze (default: 'MNI152')
        prob_threshold: Probability threshold for clustering (default: 0.7)
        cluster_size: Minimum cluster size threshold (default: 300)
        in_place: If None, save to derivatives/deepFCD-reporting/. 
                  If empty string, save alongside predictions in derivatives/deepFCD/.
                  If a path, save to that directory.
        
    Returns:
        Tuple of (results_dataframe, csv_path, nifti_path)
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If no clusters are found
    """
    # Normalize IDs
    subject_id = normalize_subject_id(subject_id)
    session_id = normalize_session_id(session_id)
    fullid = build_full_identifier(subject_id, session_id)
    
    # Initialize BIDS layout
    try:
        layout = BIDSLayout(derivatives_dir, validate=False)
    except Exception as e:
        raise RuntimeError(f"Error loading BIDS layout from {derivatives_dir}: {e}")
    
    # Find prediction files
    mean_file, var_file = find_prediction_files(layout, subject_id, session_id, space)
    print(f"Using mean file: {mean_file}")
    print(f"Using variance file: {var_file}")
    
    # Set up options
    options = setup_options(mean_file, prob_threshold, cluster_size)
    
    # Extract clusters
    output_scan, results = extract_clusters(fullid, mean_file, var_file, options)
    
    # Check if results is empty
    if results.empty:
        raise ValueError(
            f"No clusters found for {fullid} with p_thr={prob_threshold} and c_thr={cluster_size}"
        )
    
    # Process results
    results = compute_confidence_scores(results)
    results = annotate_anatomical_regions(results)
    results = format_results(results)
    
    # Generate output filenames
    csv_filename, nifti_filename = generate_output_filenames(
        subject_id, session_id, space, prob_threshold, cluster_size
    )
    
    # Save results
    columns = ["rank", "region", "coords", "probability", "confidence"]
    csv_path, nifti_path = save_results(
        results, output_scan, mean_file, csv_filename, nifti_filename, columns,
        derivatives_dir, subject_id, session_id, in_place
    )
    
    return results[columns], csv_path, nifti_path

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Rank clusters based on probability/size thresholding and uncertainty - BIDS compliant'
    )
    parser.add_argument('subject_id', help='Subject ID (e.g., sub-PX034 or PX034)')
    parser.add_argument(
        'derivatives_dir',
        help='Path to BIDS derivatives directory (e.g., /path/to/bids/derivatives/deepFCD)'
    )
    parser.add_argument('--session', help='Session ID (e.g., ses-01 or 01)')
    parser.add_argument(
        '--space',
        default='MNI152',
        help='Space to use for analysis (default: MNI152)'
    )
    parser.add_argument(
        '--p_thr', '--prob_thr',
        type=float,
        default=0.7,
        help='Probability threshold (default: 0.7)'
    )
    parser.add_argument(
        '--c_thr', '--clus_thr',
        type=int,
        default=300,
        help='Cluster size threshold (default: 300)'
    )
    parser.add_argument(
        '--in-place',
        nargs='?',
        const='',
        default=None,
        help='Output directory. If flag is present without value, saves alongside input predictions. '
             'If a path is provided, saves to that directory. '
             'If not provided, saves to derivatives/deepFCD-reporting/ (default)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for command-line execution."""
    args = parse_arguments()
    
    try:
        # Process subject clusters
        results, csv_path, nifti_path = process_subject_clusters(
            subject_id=args.subject_id,
            derivatives_dir=args.derivatives_dir,
            session_id=args.session,
            space=args.space,
            prob_threshold=args.p_thr,
            cluster_size=args.c_thr,
            in_place=getattr(args, 'in_place', None)
        )
        
        # Display results table
        print(tabulate(results, headers="keys", tablefmt="github", showindex=False))
        
        # Print save locations
        print(f"Results saved to: {csv_path}")
        print(f"Cluster mask saved to: {nifti_path}")
        
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(str(e))
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
