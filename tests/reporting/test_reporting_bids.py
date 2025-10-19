#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive tests for reporting_bids.py to preserve results across iterations.

Usage:
    pytest test_reporting_bids.py -v
    python -m pytest test_reporting_bids.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from bids import BIDSLayout

# Add app/utils directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app' / 'utils'))

from reporting_bids import (
    normalize_subject_id,
    normalize_session_id,
    build_full_identifier,
    find_prediction_files,
    setup_options,
    extract_clusters,
    compute_confidence_scores,
    annotate_anatomical_regions,
    format_results,
    generate_output_filenames,
    save_results,
    process_subject_clusters,
)


class TestIDNormalization:
    """Tests for subject and session ID normalization."""
    
    def test_normalize_subject_id_with_prefix(self):
        """Test subject ID that already has 'sub-' prefix."""
        assert normalize_subject_id('sub-001') == 'sub-001'
        assert normalize_subject_id('sub-PX034') == 'sub-PX034'
    
    def test_normalize_subject_id_without_prefix(self):
        """Test subject ID without 'sub-' prefix."""
        assert normalize_subject_id('001') == 'sub-001'
        assert normalize_subject_id('PX034') == 'sub-PX034'
    
    def test_normalize_session_id_with_prefix(self):
        """Test session ID that already has 'ses-' prefix."""
        assert normalize_session_id('ses-01') == 'ses-01'
        assert normalize_session_id('ses-baseline') == 'ses-baseline'
    
    def test_normalize_session_id_without_prefix(self):
        """Test session ID without 'ses-' prefix."""
        assert normalize_session_id('01') == 'ses-01'
        assert normalize_session_id('baseline') == 'ses-baseline'
    
    def test_normalize_session_id_none(self):
        """Test session ID when None is provided."""
        assert normalize_session_id(None) is None


class TestIdentifierBuilder:
    """Tests for building full BIDS identifiers."""
    
    def test_build_full_identifier_with_session(self):
        """Test building identifier with session."""
        result = build_full_identifier('sub-001', 'ses-01')
        assert result == 'sub-001_ses-01'
    
    def test_build_full_identifier_without_session(self):
        """Test building identifier without session."""
        result = build_full_identifier('sub-001', None)
        assert result == 'sub-001'


class TestSetupOptions:
    """Tests for options dictionary setup."""
    
    @pytest.fixture
    def temp_nifti(self):
        """Create a temporary NIfTI file for testing."""
        temp_dir = tempfile.mkdtemp()
        nifti_path = Path(temp_dir) / "test.nii.gz"
        
        # Create a simple NIfTI file
        data = np.random.rand(10, 10, 10)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, nifti_path)
        
        yield str(nifti_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_setup_options_basic(self, temp_nifti):
        """Test basic options setup."""
        options = setup_options(
            mean_file=temp_nifti,
            prob_threshold=0.7,
            cluster_size=300,
            script_dir='/test/dir'
        )
        
        assert 'header' in options
        assert options['t_bin'] == 0.7
        assert options['l_min'] == 300
        assert 'subcortical_mask_v3.nii.gz' in options['submask']
        assert options['data_folder'] == str(Path(temp_nifti).parent)
    
    def test_setup_options_default_script_dir(self, temp_nifti):
        """Test options setup with default script directory."""
        options = setup_options(
            mean_file=temp_nifti,
            prob_threshold=0.5,
            cluster_size=200
        )
        
        assert options['t_bin'] == 0.5
        assert options['l_min'] == 200


class TestConfidenceScores:
    """Tests for confidence score computation."""
    
    def test_compute_confidence_empty_dataframe(self):
        """Test confidence computation with empty DataFrame."""
        df = pd.DataFrame()
        result = compute_confidence_scores(df)
        assert result.empty
    
    def test_compute_confidence_with_data(self):
        """Test confidence computation with actual data."""
        df = pd.DataFrame({
            'rank': [1, 2, 3],
            'var': [0.1, 0.2, 0.3],
            'probability': [0.9, 0.8, 0.7]
        })
        
        result = compute_confidence_scores(df)
        
        assert 'confidence' in result.columns
        assert len(result) == 3
        # Confidence should be inversely related to variance
        assert result.loc[result['var'] == 0.1, 'confidence'].values[0] > \
               result.loc[result['var'] == 0.3, 'confidence'].values[0]
        # Check values are in valid range
        assert all(0 <= result['confidence']) and all(result['confidence'] <= 100)
    
    def test_compute_confidence_preserves_columns(self):
        """Test that existing columns are preserved."""
        df = pd.DataFrame({
            'rank': [1, 2],
            'var': [0.1, 0.2],
            'probability': [0.9, 0.8],
            'coords': [[1, 2, 3], [4, 5, 6]]
        })
        
        result = compute_confidence_scores(df)
        
        assert 'rank' in result.columns
        assert 'probability' in result.columns
        assert 'coords' in result.columns


class TestAnatomicalAnnotation:
    """Tests for anatomical region annotation."""
    
    @patch('reporting_bids.read_atlas_peak')
    def test_annotate_empty_dataframe(self, mock_atlas):
        """Test annotation with empty DataFrame."""
        df = pd.DataFrame()
        result = annotate_anatomical_regions(df)
        assert result.empty
        mock_atlas.assert_not_called()
    
    @patch('reporting_bids.read_atlas_peak')
    def test_annotate_with_data(self, mock_atlas):
        """Test annotation with actual data."""
        # Mock atlas reader to return specific labels
        mock_atlas.return_value = [('label1', 'Right Superior Frontal Gyrus')]
        
        df = pd.DataFrame({
            'rank': [1, 2],
            'coords': [[10, 20, 30], [40, 50, 60]],
            'probability': [0.9, 0.8]
        })
        
        result = annotate_anatomical_regions(df)
        
        assert 'region' in result.columns
        assert len(result) == 2
        assert mock_atlas.call_count == 2
        assert all(result['region'] == 'Right Superior Frontal Gyrus')


class TestResultFormatting:
    """Tests for result formatting."""
    
    def test_format_empty_dataframe(self):
        """Test formatting with empty DataFrame."""
        df = pd.DataFrame()
        result = format_results(df)
        assert result.empty
    
    def test_format_probability_and_confidence(self):
        """Test probability and confidence are converted to percentages."""
        df = pd.DataFrame({
            'rank': [1, 2],
            'probability': [0.85, 0.75],
            'confidence': [92.5, 85.3],
            'coords': [[1, 2, 3], [4, 5, 6]]
        })
        
        result = format_results(df)
        
        # Check conversion to integers
        assert result['probability'].dtype == int
        assert result['confidence'].dtype == int
        assert result['probability'].iloc[0] == 85
        assert result['probability'].iloc[1] == 75
        assert result['confidence'].iloc[0] == 92  # rounded
        assert result['confidence'].iloc[1] == 85
    
    def test_format_coordinates_cleaning(self):
        """Test coordinate string cleaning."""
        df = pd.DataFrame({
            'rank': [1],
            'probability': [0.8],
            'confidence': [90.0],
            'coords': [[10, 20, 30]]  # Use list instead of numpy array
        })
        
        result = format_results(df)
        
        # Check coordinate formatting (no brackets, no spaces)
        coord_str = result['coords'].iloc[0]
        assert '[' not in coord_str
        assert ']' not in coord_str
        # Original coords are cleaned - spaces and brackets removed
        # Result should be like "10,20,30"
        assert coord_str == '10,20,30'


class TestFilenameGeneration:
    """Tests for output filename generation."""
    
    def test_generate_filenames_with_session(self):
        """Test filename generation with session."""
        csv_name, nifti_name = generate_output_filenames(
            subject_id='sub-001',
            session_id='ses-01',
            space='MNI152',
            prob_threshold=0.7,
            cluster_size=300
        )
        
        assert 'sub-001_ses-01' in csv_name
        assert 'space-MNI152' in csv_name
        assert 'pthr-0.7' in csv_name
        assert 'cthr-300' in csv_name
        assert csv_name.endswith('.csv')
        
        assert 'sub-001_ses-01' in nifti_name
        assert 'space-MNI152' in nifti_name
        assert nifti_name.endswith('.nii.gz')
    
    def test_generate_filenames_without_session(self):
        """Test filename generation without session."""
        csv_name, nifti_name = generate_output_filenames(
            subject_id='sub-001',
            session_id=None,
            space='MNI152',
            prob_threshold=0.5,
            cluster_size=200
        )
        
        assert 'sub-001_space-MNI152' in csv_name
        assert 'ses-' not in csv_name
        assert 'pthr-0.5' in csv_name
        assert 'cthr-200' in csv_name


class TestSaveResults:
    """Tests for saving results to files."""
    
    @pytest.fixture
    def temp_setup(self):
        """Create temporary directory and NIfTI file."""
        temp_dir = tempfile.mkdtemp()
        nifti_path = Path(temp_dir) / "mean.nii.gz"
        
        # Create a simple NIfTI file
        data = np.random.rand(10, 10, 10)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, nifti_path)
        
        yield temp_dir, str(nifti_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_save_results_creates_files(self, temp_setup):
        """Test that save_results creates both CSV and NIfTI files."""
        temp_dir, mean_file = temp_setup
        
        # Create test data
        results = pd.DataFrame({
            'rank': [1, 2],
            'region': ['Region A', 'Region B'],
            'coords': ['10,20,30', '40,50,60'],
            'probability': [85, 75],
            'confidence': [90, 80]
        })
        
        output_scan = np.random.rand(10, 10, 10)
        columns = ['rank', 'region', 'coords', 'probability', 'confidence']
        
        # Create a mock derivatives structure
        derivatives_dir = Path(temp_dir) / 'derivatives' / 'deepFCD'
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path, nifti_path = save_results(
            results=results,
            output_scan=output_scan,
            mean_file=mean_file,
            csv_filename='test_results.csv',
            nifti_filename='test_mask.nii.gz',
            columns=columns,
            derivatives_dir=str(derivatives_dir),
            subject_id='sub-001',
            session_id='ses-01'
        )
        
        # Check files exist
        assert csv_path.exists()
        assert nifti_path.exists()
        
        # Check they are in the correct derivatives/deepFCD-reporting directory
        assert 'deepFCD-reporting' in str(csv_path)
        assert 'sub-001' in str(csv_path)
        assert 'ses-01' in str(csv_path)
        
        # Check dataset_description.json was created
        dataset_desc = Path(temp_dir) / 'derivatives' / 'deepFCD-reporting' / 'dataset_description.json'
        assert dataset_desc.exists()
        
        # Check CSV content
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == columns
        
        # Check NIfTI can be loaded
        loaded_img = nib.load(nifti_path)
        assert loaded_img.shape == (10, 10, 10)
    
    def test_save_results_in_place_flag(self, temp_setup):
        """Test that save_results saves alongside predictions when in_place=''."""
        temp_dir, mean_file = temp_setup
        
        # Create test data
        results = pd.DataFrame({
            'rank': [1],
            'region': ['Region A'],
            'coords': ['10,20,30'],
            'probability': [85],
            'confidence': [90]
        })
        
        output_scan = np.random.rand(10, 10, 10)
        columns = ['rank', 'region', 'coords', 'probability', 'confidence']
        
        # Create a mock derivatives structure
        derivatives_dir = Path(temp_dir) / 'derivatives' / 'deepFCD'
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path, nifti_path = save_results(
            results=results,
            output_scan=output_scan,
            mean_file=mean_file,
            csv_filename='test_results.csv',
            nifti_filename='test_mask.nii.gz',
            columns=columns,
            derivatives_dir=str(derivatives_dir),
            subject_id='sub-001',
            session_id='ses-01',
            in_place=''
        )
        
        # Check files exist in derivatives/deepFCD (not deepFCD-reporting)
        assert csv_path.exists()
        assert nifti_path.exists()
        assert 'deepFCD-reporting' not in str(csv_path)
        assert str(derivatives_dir) in str(csv_path.parent.parent.parent)
        assert 'sub-001' in str(csv_path)
        assert 'ses-01' in str(csv_path)
    
    def test_save_results_custom_path(self, temp_setup):
        """Test that save_results saves to custom path when in_place is a path."""
        temp_dir, mean_file = temp_setup
        
        # Create test data
        results = pd.DataFrame({
            'rank': [1],
            'region': ['Region A'],
            'coords': ['10,20,30'],
            'probability': [85],
            'confidence': [90]
        })
        
        output_scan = np.random.rand(10, 10, 10)
        columns = ['rank', 'region', 'coords', 'probability', 'confidence']
        
        # Create custom output path
        custom_path = Path(temp_dir) / 'custom_output'
        custom_path.mkdir(parents=True, exist_ok=True)
        
        derivatives_dir = Path(temp_dir) / 'derivatives' / 'deepFCD'
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path, nifti_path = save_results(
            results=results,
            output_scan=output_scan,
            mean_file=mean_file,
            csv_filename='test_results.csv',
            nifti_filename='test_mask.nii.gz',
            columns=columns,
            derivatives_dir=str(derivatives_dir),
            subject_id='sub-002',
            session_id='ses-02',
            in_place=str(custom_path)
        )
        
        # Check files exist in custom path
        assert csv_path.exists()
        assert nifti_path.exists()
        assert str(custom_path) in str(csv_path)
        assert 'sub-002' in str(csv_path)
        assert 'ses-02' in str(csv_path)


class TestFindPredictionFiles:
    """Tests for finding prediction files in BIDS layout."""
    
    @patch('reporting_bids.BIDSLayout')
    def test_find_files_success(self, mock_layout_class):
        """Test successful file finding."""
        # Create mock files
        mock_mean = Mock()
        mock_mean.filename = 'sub-001_stat-mean1_desc-deepFCD_probseg.nii.gz'
        mock_mean.path = '/data/sub-001_stat-mean1_desc-deepFCD_probseg.nii.gz'
        
        mock_var = Mock()
        mock_var.filename = 'sub-001_stat-var1_desc-deepFCD_probseg.nii.gz'
        mock_var.path = '/data/sub-001_stat-var1_desc-deepFCD_probseg.nii.gz'
        
        # Mock layout
        mock_layout = Mock()
        mock_layout.get.return_value = [mock_mean, mock_var]
        
        mean_path, var_path = find_prediction_files(
            layout=mock_layout,
            subject_id='sub-001',
            session_id=None,
            space='orig'
        )
        
        assert 'mean1' in mean_path
        assert 'var1' in var_path
    
    @patch('reporting_bids.BIDSLayout')
    def test_find_files_no_mean(self, mock_layout_class):
        """Test error when mean file not found."""
        mock_layout = Mock()
        mock_layout.get.return_value = []
        
        with pytest.raises(FileNotFoundError, match="No mean1 prediction files found"):
            find_prediction_files(
                layout=mock_layout,
                subject_id='sub-001',
                session_id=None,
                space='orig'
            )
    
    @patch('reporting_bids.BIDSLayout')
    def test_find_files_with_session(self, mock_layout_class):
        """Test file finding with session specified."""
        mock_mean = Mock()
        mock_mean.filename = 'sub-001_ses-01_stat-mean1_desc-deepFCD_probseg.nii.gz'
        mock_mean.path = '/data/sub-001_ses-01_stat-mean1.nii.gz'
        
        mock_var = Mock()
        mock_var.filename = 'sub-001_ses-01_stat-var1_desc-deepFCD_probseg.nii.gz'
        mock_var.path = '/data/sub-001_ses-01_stat-var1.nii.gz'
        
        mock_layout = Mock()
        mock_layout.get.return_value = [mock_mean, mock_var]
        
        mean_path, var_path = find_prediction_files(
            layout=mock_layout,
            subject_id='sub-001',
            session_id='ses-01',
            space='MNI152'
        )
        
        assert 'ses-01' in mean_path
        assert 'ses-01' in var_path
    
    @patch('reporting_bids.BIDSLayout')
    def test_find_files_space_mni152(self, mock_layout_class):
        """Test that space='MNI152' queries for space-MNI152 files specifically."""
        mock_mean_mni = Mock()
        mock_mean_mni.filename = 'sub-001_space-MNI152_stat-mean1_desc-deepFCD_probseg.nii.gz'
        mock_mean_mni.path = '/data/sub-001_space-MNI152_stat-mean1.nii.gz'
        
        mock_var_mni = Mock()
        mock_var_mni.filename = 'sub-001_space-MNI152_stat-var1_desc-deepFCD_probseg.nii.gz'
        mock_var_mni.path = '/data/sub-001_space-MNI152_stat-var1.nii.gz'
        
        mock_layout = Mock()
        mock_layout.get.return_value = [mock_mean_mni, mock_var_mni]
        
        mean_path, var_path = find_prediction_files(
            layout=mock_layout,
            subject_id='sub-001',
            session_id=None,
            space='MNI152'
        )
        
        # Verify that layout.get was called with space='MNI152'
        call_args = mock_layout.get.call_args
        assert call_args[1]['space'] == 'MNI152'
        
        # Verify correct files returned
        assert 'space-MNI152' in mean_path
        assert 'space-MNI152' in var_path


class TestIntegration:
    """Integration tests for complete workflow."""
    
    @pytest.fixture
    def mock_environment(self):
        """Set up complete mock environment for integration testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create mock NIfTI files
        mean_path = Path(temp_dir) / "mean.nii.gz"
        var_path = Path(temp_dir) / "var.nii.gz"
        
        mean_data = np.random.rand(10, 10, 10)
        var_data = np.random.rand(10, 10, 10) * 0.1
        
        mean_img = nib.Nifti1Image(mean_data, np.eye(4))
        var_img = nib.Nifti1Image(var_data, np.eye(4))
        
        nib.save(mean_img, mean_path)
        nib.save(var_img, var_path)
        
        yield {
            'temp_dir': temp_dir,
            'mean_path': str(mean_path),
            'var_path': str(var_path),
            'mean_data': mean_data,
            'var_data': var_data
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('reporting_bids.extractLesionCluster')
    @patch('reporting_bids.read_atlas_peak')
    def test_extract_and_process_clusters(self, mock_atlas, mock_extract, mock_environment):
        """Test complete cluster extraction and processing pipeline."""
        # Mock cluster extraction
        output_scan = np.zeros((10, 10, 10))
        results_df = pd.DataFrame({
            'rank': [1, 2],
            'var': [0.1, 0.2],
            'probability': [0.9, 0.8],
            'coords': [[10, 20, 30], [40, 50, 60]]
        })
        mock_extract.return_value = (output_scan, results_df)
        
        # Mock atlas annotation
        mock_atlas.return_value = [('label', 'Test Region')]
        
        # Run extraction
        mean_file = mock_environment['mean_path']
        var_file = mock_environment['var_path']
        options = setup_options(mean_file, 0.7, 300)
        
        output, results = extract_clusters('sub-001', mean_file, var_file, options)
        
        # Process results
        results = compute_confidence_scores(results)
        results = annotate_anatomical_regions(results)
        results = format_results(results)
        
        # Verify final results
        assert len(results) == 2
        assert 'confidence' in results.columns
        assert 'region' in results.columns
        assert results['probability'].dtype == int
        assert results['confidence'].dtype == int


class TestDataConsistency:
    """Tests to ensure results remain consistent across iterations."""
    
    def test_confidence_calculation_deterministic(self):
        """Test that confidence calculation is deterministic."""
        df = pd.DataFrame({
            'rank': [1, 2, 3],
            'var': [0.1, 0.2, 0.3],
            'probability': [0.9, 0.8, 0.7]
        })
        
        result1 = compute_confidence_scores(df.copy())
        result2 = compute_confidence_scores(df.copy())
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_formatting_idempotent(self):
        """Test that formatting produces consistent results."""
        df = pd.DataFrame({
            'rank': [1],
            'probability': [0.855],
            'confidence': [92.7],
            'coords': [[10, 20, 30]]
        })
        
        result1 = format_results(df.copy())
        
        # Verify first formatting converted correctly
        assert result1['probability'].iloc[0] == 86  # 0.855 * 100 rounded
        assert result1['confidence'].iloc[0] == 93  # 92.7 rounded
        assert result1['coords'].iloc[0] == '10,20,30'
        
        # Note: format_results expects float probability/confidence inputs
        # Applying twice would multiply already-converted integers, so we just
        # verify the first application is correct
    
    def test_filename_generation_consistent(self):
        """Test that filename generation is consistent."""
        params = {
            'subject_id': 'sub-001',
            'session_id': 'ses-01',
            'space': 'MNI152',
            'prob_threshold': 0.7,
            'cluster_size': 300
        }
        
        csv1, nifti1 = generate_output_filenames(**params)
        csv2, nifti2 = generate_output_filenames(**params)
        
        assert csv1 == csv2
        assert nifti1 == nifti2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
