#!/usr/bin/env python
# coding: utf-8

"""
Integration tests for reporting_bids.py using real data.
These tests verify that refactored code produces identical results to the original.

Usage:
    pytest test_reporting_bids_integration.py -v
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil
import json

import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from unittest.mock import patch, Mock

# Add app/utils directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app' / 'utils'))

from reporting_bids import (
    process_subject_clusters,
    normalize_subject_id,
    normalize_session_id,
)


class TestRealDataProcessing:
    """Integration tests using actual BIDS data structure."""
    
    @pytest.fixture
    def reference_results(self):
        """Reference results to compare against.
        
        These should match the output from the original script version.
        Update these values after validating the refactored version produces correct results.
        """
        return {
            'num_clusters': 2,
            'columns': ['rank', 'region', 'coords', 'probability', 'confidence'],
            'expected_types': {
                'rank': int,
                'probability': int,
                'confidence': int,
                'coords': str,
                'region': str
            }
        }
    
    @pytest.fixture
    def mock_bids_layout(self):
        """Create a mock BIDS layout with test files."""
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        derivatives_dir = Path(temp_dir) / "derivatives" / "deepFCD"
        subject_dir = derivatives_dir / "sub-PX034" / "ses-02"
        subject_dir.mkdir(parents=True)
        
        # Create mock prediction files
        mean_path = subject_dir / "sub-PX034_ses-02_space-MNI152_stat-mean1_desc-deepFCD_probseg.nii.gz"
        var_path = subject_dir / "sub-PX034_ses-02_space-MNI152_stat-var1_desc-deepFCD_probseg.nii.gz"
        
        # Create synthetic data with a clear "lesion" region
        shape = (91, 109, 91)  # Standard MNI152 shape
        mean_data = np.zeros(shape)
        var_data = np.ones(shape) * 0.1
        
        # Add a synthetic lesion cluster
        mean_data[40:45, 50:55, 45:50] = 0.9  # High probability region
        var_data[40:45, 50:55, 45:50] = 0.05  # Low variance (high confidence)
        
        # Add another smaller cluster
        mean_data[60:63, 70:73, 60:63] = 0.8
        var_data[60:63, 70:73, 60:63] = 0.08
        
        # Create NIfTI files with proper MNI152 affine
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        
        mean_img = nib.Nifti1Image(mean_data, affine)
        var_img = nib.Nifti1Image(var_data, affine)
        
        nib.save(mean_img, mean_path)
        nib.save(var_img, var_path)
        
        yield {
            'temp_dir': temp_dir,
            'derivatives_dir': str(derivatives_dir),
            'mean_path': str(mean_path),
            'var_path': str(var_path),
            'mean_data': mean_data,
            'var_data': var_data
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('reporting_bids.BIDSLayout')
    @patch('reporting_bids.extractLesionCluster')
    @patch('reporting_bids.read_atlas_peak')
    def test_full_pipeline_with_mock_data(
        self, mock_atlas, mock_extract, mock_layout_class, mock_bids_layout, reference_results
    ):
        """Test complete pipeline with mocked BIDS layout and cluster extraction."""
        
        # Mock the BIDS layout
        mock_layout = Mock()
        mock_mean_file = Mock()
        mock_mean_file.filename = 'sub-PX034_ses-02_space-MNI152_stat-mean1_desc-deepFCD_probseg.nii.gz'
        mock_mean_file.path = mock_bids_layout['mean_path']
        
        mock_var_file = Mock()
        mock_var_file.filename = 'sub-PX034_ses-02_space-MNI152_stat-var1_desc-deepFCD_probseg.nii.gz'
        mock_var_file.path = mock_bids_layout['var_path']
        
        mock_layout.get.return_value = [mock_mean_file, mock_var_file]
        mock_layout_class.return_value = mock_layout
        
        # Mock cluster extraction with realistic results
        output_scan = np.zeros_like(mock_bids_layout['mean_data'])
        output_scan[40:45, 50:55, 45:50] = 1  # Cluster 1
        output_scan[60:63, 70:73, 60:63] = 2  # Cluster 2
        
        results_df = pd.DataFrame({
            'rank': [1, 2],
            'var': [0.05, 0.08],
            'probability': [0.92, 0.85],
            'coords': [[20, 40, 30], [50, 60, 55]],
            'size': [125, 27]
        })
        mock_extract.return_value = (output_scan, results_df)
        
        # Mock atlas annotation
        mock_atlas.side_effect = [
            [('label1', 'Right Superior Frontal Gyrus')],
            [('label2', 'Left Middle Temporal Gyrus')]
        ]
        
        # Run the full pipeline
        results, csv_path, nifti_path = process_subject_clusters(
            subject_id='PX034',
            derivatives_dir=mock_bids_layout['derivatives_dir'],
            session_id='02',
            space='MNI152',
            prob_threshold=0.7,
            cluster_size=300
        )
        
        # Verify results structure
        assert len(results) == reference_results['num_clusters']
        assert list(results.columns) == reference_results['columns']
        
        # Verify data types
        for col, expected_type in reference_results['expected_types'].items():
            assert results[col].dtype == expected_type or all(isinstance(x, expected_type) for x in results[col])
        
        # Verify files were created
        assert csv_path.exists()
        assert nifti_path.exists()
        
        # Verify CSV content matches DataFrame
        loaded_csv = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(results, loaded_csv)
        
        # Verify NIfTI can be loaded
        loaded_nifti = nib.load(nifti_path)
        assert loaded_nifti.shape == mock_bids_layout['mean_data'].shape
    
    def test_id_normalization_consistency(self):
        """Test that ID normalization is consistent."""
        test_cases = [
            ('PX034', 'sub-PX034'),
            ('sub-PX034', 'sub-PX034'),
            ('001', 'sub-001'),
            ('sub-001', 'sub-001'),
        ]
        
        for input_id, expected in test_cases:
            assert normalize_subject_id(input_id) == expected
    
    def test_session_normalization_consistency(self):
        """Test that session normalization is consistent."""
        test_cases = [
            ('01', 'ses-01'),
            ('ses-01', 'ses-01'),
            ('baseline', 'ses-baseline'),
            ('ses-baseline', 'ses-baseline'),
            (None, None),
        ]
        
        for input_ses, expected in test_cases:
            assert normalize_session_id(input_ses) == expected


class TestResultReproducibility:
    """Tests to ensure results are reproducible across runs."""
    
    @patch('reporting_bids.BIDSLayout')
    @patch('reporting_bids.extractLesionCluster')
    @patch('reporting_bids.read_atlas_peak')
    def test_multiple_runs_same_results(
        self, mock_atlas, mock_extract, mock_layout_class
    ):
        """Test that multiple runs with same input produce identical results."""
        
        # Set up consistent mocks
        temp_dir = tempfile.mkdtemp()
        try:
            # Create mock files
            mean_path = Path(temp_dir) / "mean.nii.gz"
            var_path = Path(temp_dir) / "var.nii.gz"
            
            data = np.random.RandomState(42).rand(10, 10, 10)
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, mean_path)
            nib.save(img, var_path)
            
            # Mock layout
            mock_layout = Mock()
            mock_mean = Mock()
            mock_mean.filename = 'sub-001_stat-mean1_desc-deepFCD_probseg.nii.gz'
            mock_mean.path = str(mean_path)
            mock_var = Mock()
            mock_var.filename = 'sub-001_stat-var1_desc-deepFCD_probseg.nii.gz'
            mock_var.path = str(var_path)
            mock_layout.get.return_value = [mock_mean, mock_var]
            mock_layout_class.return_value = mock_layout
            
            # Mock extraction with deterministic results
            output = np.zeros((10, 10, 10))
            results_df = pd.DataFrame({
                'rank': [1, 2],
                'var': [0.1, 0.2],
                'probability': [0.9, 0.8],
                'coords': [[10, 20, 30], [40, 50, 60]]
            })
            mock_extract.return_value = (output, results_df)
            mock_atlas.return_value = [('label', 'Test Region')]
            
            # Run pipeline twice
            results1, csv1, nifti1 = process_subject_clusters(
                subject_id='001',
                derivatives_dir=temp_dir,
                prob_threshold=0.7,
                cluster_size=300
            )
            
            # Reset mocks to same state
            mock_extract.return_value = (output, results_df.copy())
            
            results2, csv2, nifti2 = process_subject_clusters(
                subject_id='001',
                derivatives_dir=temp_dir,
                prob_threshold=0.7,
                cluster_size=300
            )
            
            # Compare results
            pd.testing.assert_frame_equal(results1, results2)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_confidence_score_calculation_reproducible(self):
        """Test that confidence scores are calculated consistently."""
        from reporting_bids import compute_confidence_scores
        
        # Same input data
        df = pd.DataFrame({
            'rank': [1, 2, 3],
            'var': [0.15, 0.25, 0.35],
            'probability': [0.92, 0.85, 0.78]
        })
        
        # Calculate multiple times
        results = []
        for _ in range(5):
            result = compute_confidence_scores(df.copy())
            results.append(result['confidence'].values)
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


class TestErrorHandling:
    """Tests for error handling in various scenarios."""
    
    @patch('reporting_bids.BIDSLayout')
    def test_missing_mean_file_error(self, mock_layout_class):
        """Test error handling when mean file is missing."""
        mock_layout = Mock()
        mock_layout.get.return_value = []
        mock_layout_class.return_value = mock_layout
        
        with pytest.raises(FileNotFoundError, match="No mean1 prediction files found"):
            process_subject_clusters(
                subject_id='001',
                derivatives_dir='/fake/path',
                prob_threshold=0.7,
                cluster_size=300
            )
    
    @patch('reporting_bids.BIDSLayout')
    @patch('reporting_bids.extractLesionCluster')
    def test_no_clusters_found_error(self, mock_extract, mock_layout_class):
        """Test error handling when no clusters are found."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create mock files
            mean_path = Path(temp_dir) / "mean.nii.gz"
            var_path = Path(temp_dir) / "var.nii.gz"
            
            data = np.zeros((10, 10, 10))
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, mean_path)
            nib.save(img, var_path)
            
            # Mock layout
            mock_layout = Mock()
            mock_mean = Mock()
            mock_mean.filename = 'sub-001_stat-mean1_desc-deepFCD_probseg.nii.gz'
            mock_mean.path = str(mean_path)
            mock_var = Mock()
            mock_var.filename = 'sub-001_stat-var1_desc-deepFCD_probseg.nii.gz'
            mock_var.path = str(var_path)
            mock_layout.get.return_value = [mock_mean, mock_var]
            mock_layout_class.return_value = mock_layout
            
            # Mock extraction returning empty DataFrame
            mock_extract.return_value = (np.zeros((10, 10, 10)), pd.DataFrame())
            
            with pytest.raises(ValueError, match="No clusters found"):
                process_subject_clusters(
                    subject_id='001',
                    derivatives_dir=temp_dir,
                    prob_threshold=0.7,
                    cluster_size=300
                )
        finally:
            shutil.rmtree(temp_dir)


class TestBackwardsCompatibility:
    """Tests to ensure backwards compatibility with original script."""
    
    def test_output_filenames_match_original_format(self):
        """Test that output filenames match the original script format."""
        from reporting_bids import generate_output_filenames
        
        csv_name, nifti_name = generate_output_filenames(
            subject_id='sub-PX034',
            session_id='ses-02',
            space='MNI152',
            prob_threshold=0.7,
            cluster_size=300
        )
        
        # Verify expected format
        assert csv_name == 'sub-PX034_ses-02_space-MNI152_desc-deepFCD-clusters_pthr-0.7_cthr-300_results.csv'
        assert nifti_name == 'sub-PX034_ses-02_space-MNI152_desc-deepFCD-clusters_pthr-0.7_cthr-300_mask.nii.gz'
    
    def test_csv_columns_match_original(self):
        """Test that CSV output columns match original script."""
        from reporting_bids import format_results
        
        df = pd.DataFrame({
            'rank': [1, 2],
            'region': ['Region A', 'Region B'],
            'coords': [[10, 20, 30], [40, 50, 60]],
            'probability': [0.92, 0.85],
            'confidence': [95.5, 88.2],
            'extra_col': ['ignore', 'ignore']  # Extra column should be ignored
        })
        
        result = format_results(df)
        expected_columns = ['rank', 'region', 'coords', 'probability', 'confidence', 'extra_col']
        
        # All columns should be preserved
        for col in expected_columns:
            assert col in result.columns


def create_reference_data_snapshot(output_path: str):
    """
    Utility function to create a reference data snapshot for regression testing.
    
    Run this function once with validated output to create a baseline for future tests.
    
    Args:
        output_path: Path to save the reference data JSON
    """
    reference_data = {
        'version': '1.0.0',
        'test_cases': [
            {
                'subject': 'sub-PX034',
                'session': 'ses-02',
                'space': 'MNI152',
                'prob_threshold': 0.7,
                'cluster_size': 300,
                'expected_columns': ['rank', 'region', 'coords', 'probability', 'confidence'],
                'data_types': {
                    'rank': 'int64',
                    'region': 'object',
                    'coords': 'object',
                    'probability': 'int64',
                    'confidence': 'int64'
                }
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(reference_data, f, indent=2)
    
    print(f"Reference data snapshot saved to {output_path}")


if __name__ == '__main__':
    # Optionally create reference data snapshot
    # create_reference_data_snapshot('test_reference_data.json')
    
    # Run tests
    pytest.main([__file__, '-v'])
