#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple

import bids
import numpy as np
import setproctitle as spt
from bids import BIDSLayout
from config.experiment import options
from preprocess_bids import preprocess_image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils.env import set_theano_flags

# Import diagnostic utilities
try:
    from utils.input_diagnostic import DeepFCDInputDiagnostic
except ImportError:
    print("Warning: Could not import input diagnostic utilities")
    DeepFCDInputDiagnostic = None

# Note: Keras imports are done dynamically after Theano flags are set
# Global variables that will be set after environment setup
K = None
load_model = None
off_the_shelf_model = None
test_model = None
transform_img = None

warnings.filterwarnings("ignore")


class DeepFCDInference:
    """Deep learning-based FCD (Focal Cortical Dysplasia) inference on BIDS datasets."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the DeepFCD inference processor.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.cwd = os.path.realpath(os.path.dirname(__file__))
        self.modalities = ["T1", "FLAIR"]
        self.model = None
        self.orig_ds = None
        self.proc_ds = None
        self.outdir = None

        # Configure logging
        self._setup_logging()

        # Configure Keras backend
        os.environ["KERAS_BACKEND"] = "theano"

        # Set up paths and directories
        self._setup_paths()

        # Load BIDS datasets
        self._load_datasets()

    def _setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.DEBUG,
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
            format="{asctime} {levelname} {filename}:{lineno}: {message}",
        )

    def _setup_paths(self):
        """Set up input and output paths."""
        # Debug logging to track None values
        logging.info("Setting up paths...")
        logging.info(f"args.bidspath: {self.args.bidspath} (type: {type(self.args.bidspath)})")
        logging.info(f"args.outpath: {self.args.outpath} (type: {type(self.args.outpath)})")
        
        if self.args.bidspath is None:
            raise ValueError("bidspath argument is None")
        
        if not os.path.isabs(self.args.bidspath):
            self.args.bidspath = os.path.abspath(self.args.bidspath)

        if self.args.outpath == "":
            # Use BIDS root as default and create derivatives/deepFCD structure
            self.outdir = os.path.join(self.args.bidspath, "derivatives", "deepFCD")
        else:
            self.outdir = self.args.outpath

        # Set up preprocessing output directory (separate from inference)
        self.preproc_outdir = os.path.join(
            self.args.bidspath, "derivatives", "deepFCD-preproc"
        )
        
        # Debug logging for final paths
        logging.info(f"Final outdir: {self.outdir} (type: {type(self.outdir)})")
        logging.info(f"Final preproc_outdir: {self.preproc_outdir} (type: {type(self.preproc_outdir)})")

        # Create both directories
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.preproc_outdir, exist_ok=True)

        # Create dataset descriptions for both
        self._create_dataset_description()
        self._create_preproc_dataset_description()

    def _create_dataset_description(self):
        """Create BIDS dataset description file for derivatives."""
        dataset_description = {
            "Name": "deepFCD",
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "deepFCD",
                    "Version": "1.0.0",
                    "Description": "Deep learning-based Focal Cortical Dysplasia detection",
                    "CodeURL": "https://github.com/NOEL-MNI/deepFCD",
                }
            ],
            "HowToAcknowledge": "Please cite the deepFCD paper when using this software.",
            "PipelineDescription": {
                "Name": "deepFCD",
                "Version": "1.0.0",
                "Description": "Automated detection of Focal Cortical Dysplasia using deep learning",
            },
        }

        import json

        with open(os.path.join(self.outdir, "dataset_description.json"), "w") as f:
            json.dump(dataset_description, f, indent=2)

    def _create_preproc_dataset_description(self):
        """Create BIDS dataset description file for preprocessing derivatives."""
        dataset_description = {
            "Name": "deepFCD-preproc",
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "deepFCD-preproc",
                    "Version": "1.0.0",
                    "Description": "Preprocessing pipeline for deepFCD: brain extraction, registration, and bias correction",
                    "CodeURL": "https://github.com/NOEL-MNI/deepFCD",
                }
            ],
            "HowToAcknowledge": "Please cite the deepFCD paper when using this software.",
            "PipelineDescription": {
                "Name": "deepFCD-preproc",
                "Version": "1.0.0",
                "Description": "Preprocessing steps including brain extraction, MNI152 registration, and N3 bias correction",
            },
        }

        import json

        with open(
            os.path.join(self.preproc_outdir, "dataset_description.json"), "w"
        ) as f:
            json.dump(dataset_description, f, indent=2)

    def _load_datasets(self):
        """Load original and processed BIDS datasets."""
        print(self.args.bidspath)
        self.orig_ds = BIDSLayout(self.args.bidspath, validate=False)
        print(self.orig_ds)

    def get_subjects(self, dataset: BIDSLayout) -> List[str]:
        """Get list of subjects to process.

        Args:
            dataset: BIDS dataset layout

        Returns:
            List of subject IDs
        """
        if self.args.subjects is None:
            subjects = dataset.get_subjects()
        else:
            subjects = [s.replace("sub-", "") for s in self.args.subjects]
            print(subjects)
        return subjects

    def get_subject_sessions(self, dataset: BIDSLayout) -> Dict[str, List[str]]:
        """Get sessions for each subject.

        Args:
            dataset: BIDS dataset layout

        Returns:
            Dictionary mapping subject IDs to list of session IDs
        """
        subjects = self.get_subjects(dataset)
        subject_sessions = {}

        for subject in subjects:
            sessions = dataset.get_sessions(subject=subject)
            if sessions:
                # If sessions exist, use them
                subject_sessions[subject] = sessions
            else:
                # If no sessions, use None to indicate no session structure
                subject_sessions[subject] = [None]

        return subject_sessions


class DeepFCDPreprocessor:
    """Handle preprocessing of BIDS images for DeepFCD inference."""

    def __init__(self, inference_processor: DeepFCDInference):
        """Initialize preprocessor.

        Args:
            inference_processor: Main inference processor instance
        """
        self.inference = inference_processor

    def _check_preprocessing_outputs_exist(self, fullid: str) -> bool:
        """Check if preprocessing outputs already exist for a subject.

        Args:
            fullid: Full subject ID (e.g., sub-001 or sub-001_ses-01)

        Returns:
            True if all expected preprocessing outputs exist, False otherwise
        """
        # Construct expected output directory
        if '_ses-' in fullid:
            subject_part, session_part = fullid.split('_ses-', 1)
            output_dir = os.path.join(self.inference.preproc_outdir, subject_part, f'ses-{session_part}', 'preproc')
        else:
            output_dir = os.path.join(self.inference.preproc_outdir, fullid, 'preproc')

        # Expected output files
        expected_files = [
            os.path.join(output_dir, f"{fullid}_space-MNI152_T1w_brain.nii.gz"),
            os.path.join(output_dir, f"{fullid}_space-MNI152_FLAIR_brain.nii.gz"),
        ]

        # Check if all files exist
        all_exist = all(os.path.isfile(f) for f in expected_files)
        
        if all_exist:
            logging.info(f"Preprocessing outputs already exist for {fullid}")
            for f in expected_files:
                logging.debug(f"  Found: {f}")
        else:
            missing_files = [f for f in expected_files if not os.path.isfile(f)]
            logging.debug(f"Missing preprocessing outputs for {fullid}: {len(missing_files)} files")
            for f in missing_files:
                logging.debug(f"  Missing: {f}")
        
        return all_exist

    def _provide_diagnostic_guidance(self, fullid: str, missing_modality: str) -> None:
        """Provide detailed guidance when required files are not found.
        
        Args:
            fullid: Full subject ID (e.g., sub-001_ses-01)
            missing_modality: The modality that's missing (T1w or FLAIR)
        """
        print("\n" + "="*70)
        print(f"🔍 DEEPFCD INPUT DIAGNOSTIC for {fullid}")
        print("="*70)
        
        # Parse subject and session
        if '_ses-' in fullid:
            subject_id, session_part = fullid.split('_ses-', 1)
            session_id = f"ses-{session_part}"
        else:
            subject_id, session_id = fullid, None
            
        # Expected file patterns
        expected_t1 = f"{fullid}_space-MNI152_T1w_brain.nii.gz"
        expected_t2 = f"{fullid}_space-MNI152_FLAIR_brain.nii.gz"
        
        # Expected directory
        if session_id:
            expected_dir = os.path.join(
                self.inference.preproc_outdir, subject_id, session_id, 'preproc'
            )
        else:
            expected_dir = os.path.join(self.inference.preproc_outdir, fullid, 'preproc')
            
        print(f"❌ Missing: {missing_modality} preprocessed files for {fullid}")
        print("📁 Expected location: {expected_dir}")
        print("📄 Expected files:")
        print(f"   • {expected_t1}")
        print(f"   • {expected_t2}")
        
        # Check what's actually available
        print("\n🔍 Checking available files...")
        
        # Check if directory exists
        if os.path.exists(expected_dir):
            files_in_dir = os.listdir(expected_dir)
            if files_in_dir:
                print(f"✓ Directory exists with {len(files_in_dir)} files:")
                for f in sorted(files_in_dir):
                    if f.endswith('.nii.gz'):
                        print(f"   📄 {f}")
            else:
                print(f"⚠️  Directory exists but is empty")
        else:
            print(f"❌ Expected directory does not exist: {expected_dir}")
            
        # Check for raw files in original dataset
        self._check_raw_files_availability(subject_id, session_id)
        
        # Provide recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Check if raw files are available
        raw_available = self._check_if_raw_files_exist(subject_id, session_id)
        
        if raw_available["has_t1"] and raw_available["has_flair"]:
            print("🔧 SOLUTION: Raw T1w and FLAIR images found. Run preprocessing:")
            print("")
            print("   python app/inference_bids.py \\")
            print(f"       -bp {self.inference.args.bids_path} \\")
            print(f"       -sp {options['MNI152space']} \\")
            print("       -pp \\")
            print("       -bm")
            print("")
            print("   The -pp flag will automatically generate the required files.")
            
        elif raw_available["has_t1"] or raw_available["has_flair"]:
            print("⚠️  PARTIAL DATA: Only some raw images found")
            missing_modalities = []
            if not raw_available["has_t1"]:
                missing_modalities.append("T1w")
            if not raw_available["has_flair"]:
                missing_modalities.append("FLAIR")
                
            print(f"   Missing: {', '.join(missing_modalities)}")
            print("   📋 Check your BIDS dataset structure")
            
        else:
            print("❌ NO RAW DATA: No T1w or FLAIR images found in original dataset")
            print("   📋 Required BIDS structure:")
            print(f"   {self.inference.args.bids_path}/")
            if session_id:
                print(f"   ├── {subject_id}/")
                print(f"   │   └── {session_id}/")
                print("   │       └── anat/")
                print(f"   │           ├── {fullid}_T1w.nii.gz")
                print(f"   │           └── {fullid}_FLAIR.nii.gz")
            else:
                print(f"   ├── {subject_id}/")
                print("   │   └── anat/")
                print(f"   │       ├── {fullid}_T1w.nii.gz")
                print(f"   │       └── {fullid}_FLAIR.nii.gz")
                
        print("\n📚 For more information:")
        print("   • Input requirements: docs/input_requirements.md")
        print("   • BIDS specification: https://bids-specification.readthedocs.io/")
        print("="*70 + "\n")
        
    def _check_raw_files_availability(self, subject_id: str, session_id: Optional[str]) -> None:
        """Check and report on raw file availability."""
        if not self.inference.orig_ds:
            print(f"⚠️  Cannot check raw files (original dataset not loaded)")
            return
            
        # Build query
        query = {
            "subject": subject_id.replace('sub-', ''),
            "extension": [".nii.gz", ".nii"]
        }
        if session_id:
            query["session"] = session_id.replace('ses-', '')
            
        # Check T1w
        t1_files = self.inference.orig_ds.get(suffix="T1w", **query)
        flair_files = self.inference.orig_ds.get(suffix="FLAIR", **query)
        
        print("📊 Raw file availability:")
        if t1_files:
            print(f"   ✓ T1w: {t1_files[0].path}")
        else:
            print("   ❌ T1w: Not found")
            
        if flair_files:
            print(f"   ✓ FLAIR: {flair_files[0].path}")
        else:
            print("   ❌ FLAIR: Not found")
            
    def _check_if_raw_files_exist(self, subject_id: str, session_id: Optional[str]) -> Dict[str, bool]:
        """Check if raw files exist and return status."""
        result = {"has_t1": False, "has_flair": False}
        
        if not self.inference.orig_ds:
            return result
            
        # Build query
        query = {
            "subject": subject_id.replace('sub-', ''),
            "extension": [".nii.gz", ".nii"]
        }
        if session_id:
            query["session"] = session_id.replace('ses-', '')
            
        # Check files
        t1_files = self.inference.orig_ds.get(suffix="T1w", **query)
        flair_files = self.inference.orig_ds.get(suffix="FLAIR", **query)
        
        result["has_t1"] = len(t1_files) > 0
        result["has_flair"] = len(flair_files) > 0
        
        return result

    def preprocess_subjects(self, subjects: List[str]):
        """Preprocess images for all subjects if brain masking is requested.

        Args:
            subjects: List of subject IDs to preprocess
        """
        if not self.inference.args.brainmask:
            logging.info(
                "Skipping image preprocessing and brain masking, presumably images are co-registered, bias-corrected, and skull-stripped"
            )
            return

        # Get subject-session mapping
        subject_sessions = self.inference.get_subject_sessions(self.inference.orig_ds)

        # Prepare file paths for multiprocessing
        t1w_paths = []
        flair_paths = []
        fullids = []
        skipped_subjects = []

        for subject in subjects:
            sessions = subject_sessions.get(subject, [None])
            logging.info(f"Subject {subject} has sessions: {sessions}")
            for session in sessions:
                # Build query parameters
                query_params = {
                    "subject": subject,
                    "suffix": "T1w",
                    "extension": ".nii.gz",
                }
                if session is not None:
                    query_params["session"] = session

                try:
                    t1w_file = self.inference.orig_ds.get(**query_params)[0]

                    # Update query for FLAIR
                    query_params["suffix"] = "FLAIR"
                    flair_file = self.inference.orig_ds.get(**query_params)[0]

                    # Create appropriate full ID
                    if session is not None:
                        fullid = f"sub-{subject}_ses-{session}"
                    else:
                        fullid = f"sub-{subject}"
                    
                    # Check if preprocessing outputs already exist
                    if self._check_preprocessing_outputs_exist(fullid):
                        if self.inference.args.overwrite_preprocessing:
                            logging.info(f"Preprocessing outputs exist for {fullid}, but --overwrite-preprocessing specified")
                        else:
                            logging.info(f"Skipping preprocessing for {fullid} (outputs already exist, use --overwrite-preprocessing to force)")
                            skipped_subjects.append(fullid)
                            continue

                    t1w_paths.append(t1w_file.path)
                    flair_paths.append(flair_file.path)
                    fullids.append(fullid)
                    logging.info(f"Successfully found files for {fullid}")

                except (IndexError, AttributeError) as e:
                    session_str = f"_ses-{session}" if session else ""
                    logging.warning(
                        f"Could not find files for sub-{subject}{session_str}: {e}"
                    )
                    continue

        use_gpu = self.inference.args.device.startswith("cuda")

        # Log summary of what will be processed vs skipped
        total_subjects = len(fullids) + len(skipped_subjects)
        logging.info("Preprocessing summary:")
        logging.info(f"  Total subject-session combinations found: {total_subjects}")
        logging.info(f"  Will be processed: {len(fullids)}")
        logging.info(f"  Skipped (outputs exist): {len(skipped_subjects)}")
        
        if skipped_subjects:
            logging.info("Skipped subjects:")
            for i, fullid in enumerate(skipped_subjects):
                logging.info(f"  {i+1}. {fullid}")
        
        if fullids:
            logging.info("Will process:")
            for i, fullid in enumerate(fullids):
                logging.info(f"  {i+1}. {fullid}")
        
        if not fullids:
            if skipped_subjects:
                logging.info("All preprocessing outputs already exist. Use --overwrite to force reprocessing.")
            else:
                logging.warning("No valid subject-session combinations found for preprocessing")
            return

        # Process images in parallel using the preprocessing derivatives directory
        # Use max_workers=1 to avoid confusion with multiple parallel processes
        process_map(
            partial(
                preprocess_image,
                indir_=self.inference.args.bidspath,
                outdir_=self.inference.preproc_outdir,
                preprocess=self.inference.args.preprocess,
                use_gpu=use_gpu,
            ),
            fullids,
            t1w_paths,
            flair_paths,
            max_workers=4,
        )


class DeepFCDModel:
    """Handle model loading and configuration for DeepFCD inference."""

    def __init__(self, inference_processor: DeepFCDInference):
        """Initialize model handler.

        Args:
            inference_processor: Main inference processor instance
        """
        self.inference = inference_processor
        self.model = None

    def setup_environment(self):
        """Set up Theano environment and import dependencies."""
        # Set THEANO_FLAGS based on the device argument
        set_theano_flags(self.inference.args.device)
        logging.info(os.environ["THEANO_FLAGS"])

        # Import keras after theano setup
        os.environ["KERAS_BACKEND"] = "theano"

        # These imports need to be done after Keras backend is set
        global K, load_model, off_the_shelf_model, test_model, transform_img
        from keras import backend as K
        from keras.models import load_model
        from models.noel_models_keras import off_the_shelf_model
        from utils.base import test_model, transform_img

        # Store references for later use
        self.K = K
        self.load_model = load_model

        return K, load_model

    def configure_model_options(self):
        """Configure model options and parameters."""
        # deepFCD configuration
        self.K.set_image_dim_ordering("th")
        self.K.set_image_data_format(
            "channels_first"
        )  # TH dimension ordering in this code

        options["parallel_gpu"] = False

        # Model configuration
        options["dropout_mc"] = True
        options["batch_size"] = 350000
        options["mini_batch_size"] = 2048
        options["load_checkpoint_1"] = True
        options["load_checkpoint_2"] = True

        # Set paths and experiment name
        options["test_folder"] = self.inference.outdir
        options["weight_paths"] = os.path.join(self.inference.cwd, "weights")
        options["experiment"] = "noel_deepFCD_dropoutMC"

        logging.info("experiment: {}".format(options["experiment"]))
        logging.info(f"test_folder: {options['test_folder']}")
        logging.info(f"outdir: {self.inference.outdir}")
        logging.info(f"bidspath: {self.inference.args.bidspath}")

        options['MNI152space'] = "MNI152"
        
        # Validate critical paths
        if options["test_folder"] is None:
            raise ValueError("test_folder is None - check outdir configuration")
        if not isinstance(options["test_folder"], (str, bytes, os.PathLike)):
            raise ValueError(f"test_folder is not a valid path type: {type(options['test_folder'])} = {options['test_folder']}")
        spt.setproctitle(options["experiment"])

    def load_trained_model(self):
        """Load the trained CNN model weights."""
        # Initialize the CNN architecture
        self.model = off_the_shelf_model(options)

        # Load first model
        load_weights = os.path.join(
            options["weight_paths"], "noel_deepFCD_dropoutMC_model_1.h5"
        )
        (
            logging.info("loading DNN1, model[0]: {} exists".format(load_weights))
            if os.path.isfile(load_weights)
            else sys.exit("model[0]: {} doesn't exist".format(load_weights))
        )
        self.model[0] = load_model(load_weights)

        # Load second model
        load_weights = os.path.join(
            options["weight_paths"], "noel_deepFCD_dropoutMC_model_2.h5"
        )
        (
            logging.info("loading DNN2, model[1]: {} exists".format(load_weights))
            if os.path.isfile(load_weights)
            else sys.exit("model[1]: {} doesn't exist".format(load_weights))
        )
        self.model[1] = load_model(load_weights)
        logging.info(self.model[1].summary())

        return self.model


class DeepFCDProcessor:
    """Main processor for running inference on subjects."""

    def __init__(
        self, inference_processor: DeepFCDInference, model_handler: DeepFCDModel
    ):
        """Initialize processor.

        Args:
            inference_processor: Main inference processor instance
            model_handler: Model handler instance
        """
        self.inference = inference_processor
        self.model_handler = model_handler

    def process_subjects(self, subjects: List[str]):
        """Run inference on all subjects.

        Args:
            subjects: List of subject IDs to process
        """
        # Check if preprocessing output directory exists and has content
        if not os.path.exists(self.inference.preproc_outdir):
            logging.error(f"Preprocessing output directory does not exist: {self.inference.preproc_outdir}")
            logging.error("Please run preprocessing first with the -bm flag")
            return
            
        # Check if preprocessing directory has any subject folders
        subject_dirs = [d for d in os.listdir(self.inference.preproc_outdir) 
                       if os.path.isdir(os.path.join(self.inference.preproc_outdir, d)) 
                       and d.startswith('sub-')]
        
        if not subject_dirs:
            logging.error(f"No preprocessed subjects found in: {self.inference.preproc_outdir}")
            logging.error("Please run preprocessing first with the -bm flag")
            return
            
        logging.info(f"Found {len(subject_dirs)} preprocessed subjects: {subject_dirs}")

        # Load processed dataset from preprocessing derivatives
        try:
            self.inference.proc_ds = BIDSLayout(
                self.inference.preproc_outdir, validate=False
            )
            print(self.inference.proc_ds)
        except Exception as e:
            logging.error(f"Failed to create BIDSLayout for preprocessed data: {e}")
            logging.error(f"Preprocessing directory: {self.inference.preproc_outdir}")
            return

        # Get subject-session mapping from processed dataset
        subject_sessions = self.inference.get_subject_sessions(self.inference.proc_ds)

        # Import required functions after Keras setup
        # These are already imported globally in setup_environment

        # Create list of all subject-session combinations for progress tracking
        processing_items = []
        for subject in subjects:
            sessions = subject_sessions.get(subject, [None])
            for session in sessions:
                if session is not None:
                    fullid = f"sub-{subject}_ses-{session}"
                else:
                    fullid = f"sub-{subject}"
                processing_items.append((subject, session, fullid))

        # Process each subject-session combination with progress bar
        for subject_id, session_id, fullid in tqdm(
            processing_items,
            desc="serving predictions using the trained model",
            colour="blue",
        ):
            logging.info(f"Processing {fullid}")
            self._process_single_subject(
                subject_id, session_id, fullid, test_model, transform_img
            )

    def _process_single_subject(
        self,
        subject_id: str,
        session_id: Optional[str],
        fullid: str,
        test_model_func,
        transform_img_func,
    ):
        """Process a single subject (and session if applicable) for inference.

        Args:
            subject_id: Subject ID to process
            session_id: Session ID to process (None if no sessions)
            fullid: Full subject/session ID (e.g., "sub-001" or "sub-001_ses-01")
            test_model_func: Function for testing the model
            transform_img_func: Function for transforming images
        """
        options["fullid"] = fullid

        # Get file paths
        file_paths = self._get_subject_file_paths(subject_id, session_id, fullid)
        if not file_paths:
            return

        t1_file, t2_file, orig_bidsfiles, orig_files, t1_transform, t2_transform = (
            file_paths
        )

        # Prepare data structures
        files = [t1_file, t2_file]
        transform_files = [t1_transform, t2_transform]

        test_data = {fullid: {m: f for m, f in zip(self.inference.modalities, files)}}
        test_transforms = {
            fullid: {m: n for m, n in zip(self.inference.modalities, transform_files)}
        }

        t_data = {fullid: test_data[fullid]}
        transforms = {fullid: test_transforms[fullid]}

        # Set up prediction folder following BIDS derivatives structure
        # Validate test_folder before using it
        if options["test_folder"] is None:
            raise ValueError(f"test_folder is None for {fullid}")
        if not isinstance(options["test_folder"], (str, bytes, os.PathLike)):
            raise ValueError(f"test_folder is not a valid path type for {fullid}: {type(options['test_folder'])} = {options['test_folder']}")
            
        # Extract subject and session from fullid
        if "_ses-" in fullid:
            subject_part, session_part = fullid.split("_ses-", 1)
            session_folder = f"ses-{session_part}"
            pred_folder = os.path.join(
                options["test_folder"], subject_part, session_folder, "anat"
            )
        else:
            pred_folder = os.path.join(options["test_folder"], fullid, "anat")

        options["pred_folder"] = pred_folder
        os.makedirs(options["pred_folder"], exist_ok=True)

        # Check if predictions already exist
        pred_files = self._get_prediction_file_paths(fullid)
        if self._predictions_exist(
            pred_files, orig_bidsfiles, orig_files, transform_files, transform_img_func
        ):
            return

        # Run inference
        self._run_inference(
            fullid,
            t_data,
            transforms,
            orig_files,
            test_model_func,
            transform_img_func,
            orig_bidsfiles,
            transform_files,
        )

    def _get_subject_file_paths(
        self, subject_id: str, session_id: Optional[str], fullid: str
    ) -> Optional[Tuple]:
        """Get file paths for a subject (and session if applicable).

        Args:
            subject_id: Subject ID
            session_id: Session ID (None if no sessions)
            fullid: Full subject/session ID with appropriate prefix

        Returns:
            Tuple of file paths or None if files not found
        """
        try:
            # Build query parameters for processed files
            proc_query = {
                "subject": subject_id,
                "space": options['MNI152space'],
                "label": "brain",
                "extension": ".nii.gz",
            }

            # Build query parameters for original files
            orig_query = {"subject": subject_id, "extension": ".nii.gz"}

            # Build query parameters for transform files
            transform_query = {"subject": subject_id, "extension": "mat"}

            if session_id is not None:
                proc_query["session"] = session_id
                orig_query["session"] = session_id
                transform_query["session"] = session_id

            # Get T1 files - try direct file path first, then BIDS query
            bids_subject_id = f"sub-{subject_id}" if not subject_id.startswith("sub-") else subject_id
            bids_session_id = f"ses-{session_id}" if session_id and not session_id.startswith("ses-") else session_id
            
            t1_file = self._get_preprocessed_file(bids_subject_id, bids_session_id, "T1w", self.inference.preproc_outdir)
            if not t1_file:
                # Fallback to BIDS query for T1w
                proc_query["suffix"] = "T1w"
                t1_proc_files = self.inference.proc_ds.get(**proc_query)
                if not t1_proc_files:
                    logging.error(f"No processed T1w files found for {fullid}")
                    self._provide_diagnostic_guidance(bids_subject_id, bids_session_id, self.inference.preproc_outdir)
                    return None
                t1_file = t1_proc_files[0].path
                
            if t1_file is None:
                logging.error(f"T1w file path is None for {fullid}")
                return None

            orig_query["suffix"] = "T1w"
            orig_t1_files = self.inference.orig_ds.get(**orig_query)
            if not orig_t1_files:
                logging.error(f"No original T1w files found for {fullid}")
                return None
            orig_t1_file = orig_t1_files[0]

            # Get T1 transform file - try direct file path first, then BIDS query
            t1_transform = self._get_transform_file(bids_subject_id, bids_session_id, "T1w", self.inference.preproc_outdir)
            if not t1_transform:
                # Fallback to BIDS query for T1w transforms
                transform_query["suffix"] = "T1w"
                t1_transform_files = self.inference.proc_ds.get(**transform_query)
                if not t1_transform_files:
                    logging.error(f"No T1w transform files found for {fullid}")
                    return None
                t1_transform = t1_transform_files[0].path
                if t1_transform is None:
                    logging.error(f"T1w transform path is None for {fullid}")
                    return None

            # Get FLAIR files
            # Get FLAIR/T2 files - try direct file path first, then BIDS query  
            t2_file = self._get_preprocessed_file(bids_subject_id, bids_session_id, "FLAIR", self.inference.preproc_outdir)
            if not t2_file:
                # Fallback to BIDS query for FLAIR
                proc_query["suffix"] = "FLAIR"
                t2_proc_files = self.inference.proc_ds.get(**proc_query)
                if not t2_proc_files:
                    logging.error(f"No processed FLAIR files found for {fullid}")
                    self._provide_diagnostic_guidance(bids_subject_id, bids_session_id, self.inference.preproc_outdir)
                    return None
                t2_file = t2_proc_files[0].path
                
            if t2_file is None:
                logging.error(f"FLAIR file path is None for {fullid}")
                return None

            orig_query["suffix"] = "FLAIR"
            orig_t2_files = self.inference.orig_ds.get(**orig_query)
            if not orig_t2_files:
                logging.error(f"No original FLAIR files found for {fullid}")
                return None
            orig_t2_file = orig_t2_files[0]

            # Get FLAIR transform file - try direct file path first, then BIDS query
            t2_transform = self._get_transform_file(bids_subject_id, bids_session_id, "FLAIR", self.inference.preproc_outdir)
            if not t2_transform:
                # Fallback to BIDS query for FLAIR transforms
                transform_query["suffix"] = "FLAIR"
                t2_transform_files = self.inference.proc_ds.get(**transform_query)
                if not t2_transform_files:
                    logging.error(f"No FLAIR transform files found for {fullid}")
                    return None
                t2_transform = t2_transform_files[0].path
                if t2_transform is None:
                    logging.error(f"FLAIR transform path is None for {fullid}")
                    return None

            orig_bidsfiles = [orig_t1_file, orig_t2_file]
            orig_files = [bf.path for bf in orig_bidsfiles]

            return (
                t1_file,
                t2_file,
                orig_bidsfiles,
                orig_files,
                t1_transform,
                t2_transform,
            )

        except (IndexError, AttributeError, TypeError) as e:
            logging.error(f"Error getting file paths for {fullid}: {e}")
            logging.error(f"  Processed dataset directory: {self.inference.preproc_outdir}")
            logging.error(f"  Original dataset directory: {self.inference.args.bidspath}")
            # List available files for debugging
            if hasattr(self.inference, 'proc_ds') and self.inference.proc_ds:
                try:
                    available_files = self.inference.proc_ds.get(subject=subject_id)
                    logging.error(f"  Available processed files for sub-{subject_id}: {len(available_files)} files")
                    for f in available_files[:5]:  # Show first 5 files
                        logging.error(f"    {f.path}")
                    if len(available_files) > 5:
                        logging.error(f"    ... and {len(available_files) - 5} more files")
                except Exception:
                    logging.error("  Could not list available processed files")
            return None

    def _get_prediction_file_paths(self, fullid: str) -> Tuple[str, str]:
        """Get prediction file paths following BIDS derivatives conventions.

        Args:
            fullid: Full subject ID

        Returns:
            Tuple of prediction file paths (mean, variance)
        """
        # Validate pred_folder is set
        if "pred_folder" not in options:
            raise ValueError(f"pred_folder not set in options for {fullid}")
        if options["pred_folder"] is None:
            raise ValueError(f"pred_folder is None for {fullid}")
        if not isinstance(options["pred_folder"], (str, bytes, os.PathLike)):
            raise ValueError(f"pred_folder is not a valid path type for {fullid}: {type(options['pred_folder'])} = {options['pred_folder']}")
        
        # Create BIDS-compliant filenames for predictions
        base_filename = f"{fullid}_space-{options['MNI152space']}_desc-deepFCD"

        pred_mean_fname = os.path.join(
            options["pred_folder"],
            f"{base_filename}_probseg-mean.nii.gz",
        )
        pred_var_fname = os.path.join(
            options["pred_folder"],
            f"{base_filename}_probseg-var.nii.gz",
        )
        return pred_mean_fname, pred_var_fname

    def _predictions_exist(
        self,
        pred_files: Tuple[str, str],
        orig_bidsfiles,
        orig_files,
        transform_files,
        transform_img_func,
    ) -> bool:
        """Check if predictions already exist and handle accordingly.

        Args:
            pred_files: Tuple of prediction file paths
            orig_bidsfiles: Original BIDS files
            orig_files: Original file paths
            transform_files: Transform file paths
            transform_img_func: Transform function

        Returns:
            True if predictions exist and we should skip, False otherwise
        """
        pred_mean_fname, pred_var_fname = pred_files

        if np.logical_and(
            os.path.isfile(pred_mean_fname), os.path.isfile(pred_var_fname)
        ):
            logging.info("prediction for {} already exists".format(options["fullid"]))
            if not self.inference.args.overwrite:
                targetspace = "T1w"
                if "space" in orig_bidsfiles[0].entities:
                    targetspace = orig_bidsfiles[0].entities["space"]

                transform_img_func(
                    pred_mean_fname,
                    bids.layout.parse_file_entities(pred_mean_fname),
                    orig_files[0],
                    transform_files[0],
                    targetspace=targetspace,
                    invert=True,
                )
                transform_img_func(
                    pred_var_fname,
                    bids.layout.parse_file_entities(pred_var_fname),
                    orig_files[0],
                    transform_files[0],
                    targetspace=targetspace,
                    invert=True,
                )
                return True
            else:
                logging.info("overwriting...")
        return False

    def _run_inference(
        self,
        fullid: str,
        t_data: Dict,
        transforms: Dict,
        orig_files: List[str],
        test_model_func,
        transform_img_func,
        orig_bidsfiles,
        transform_files,
    ):
        """Run inference on a subject.

        Args:
            fullid: Full subject ID
            t_data: Test data dictionary
            transforms: Transform data dictionary
            orig_files: Original file paths
            test_model_func: Test model function
            transform_img_func: Transform image function
            orig_bidsfiles: Original BIDS files
            transform_files: Transform file paths
        """
        options["test_scan"] = fullid

        start = time.time()
        logging.info("\n")
        logging.info("-" * 70)
        logging.info("testing the model for scan: {}".format(fullid))
        logging.info("-" * 70)

        # Check if transforms exist
        if not any(
            [
                os.path.exists(transforms[fullid]["T1"]),
                os.path.exists(transforms[fullid]["FLAIR"]),
            ]
        ):
            transforms = None

        # Run model inference
        outputs = test_model_func(
            self.model_handler.model,
            t_data,
            options,
            performance=True,
            uncertainty=True,
            transforms=transforms,
            orig_files=orig_files,
            invert_xfrm=True,
        )

        # Transform outputs back to original space
        for k, v in outputs.items():
            targetspace = None
            if "space" in orig_bidsfiles[0].entities:
                targetspace = orig_bidsfiles[0].entities["space"]

            transform_img_func(
                v,
                bids.layout.parse_file_entities(v),
                orig_files[0],
                transform_files[0],
                targetspace=targetspace,
                invert=True,
            )

        end = time.time()
        diff = (end - start) // 60
        logging.info("-" * 70)
        logging.info("time elapsed: ~ {} minutes".format(diff))
        logging.info("-" * 70)

    def _get_preprocessed_file(self, subject_id: str, session_id: str, modality: str, preproc_outdir: str) -> Optional[str]:
        """Get preprocessed file path by checking direct file paths.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier  
            modality: Modality type (T1w or FLAIR)
            preproc_outdir: Preprocessing output directory
            
        Returns:
            Path to preprocessed file if found, None otherwise
        """
        # Determine file suffix based on modality
        if modality == "T1w":
            suffix = "space-MNI152_T1w_brain.nii.gz"
        elif modality == "FLAIR":
            suffix = "space-MNI152_FLAIR_brain.nii.gz"
        else:
            logging.warning(f"Unknown modality: {modality}")
            return None
            
        # Construct expected file path - files are in preproc subdirectory
        filename = f"{subject_id}_{session_id}_{suffix}"
        filepath = os.path.join(preproc_outdir, subject_id, session_id, "preproc", filename)
        
        # Check if file exists
        if os.path.exists(filepath):
            logging.info(f"Found preprocessed {modality} file: {filepath}")
            return filepath
        else:
            logging.debug(f"Preprocessed {modality} file not found at: {filepath}")
            return None

    def _get_transform_file(self, subject_id: str, session_id: str, modality: str, preproc_outdir: str) -> Optional[str]:
        """Get transform file path by checking direct file paths.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier  
            modality: Modality type (T1w or FLAIR)
            preproc_outdir: Preprocessing output directory
            
        Returns:
            Path to transform file if found, None otherwise
        """
        # Determine file suffix based on modality
        if modality == "T1w":
            suffix = "from-T1w_to-MNI152_fwdaffine.mat"
        elif modality == "FLAIR":
            suffix = "from-FLAIR_to-MNI152_fwdaffine.mat"
        else:
            logging.warning(f"Unknown modality: {modality}")
            return None
            
        # Construct expected file path - transform files are in preproc/transforms subdirectory
        filename = f"{subject_id}_{session_id}_{suffix}"
        filepath = os.path.join(preproc_outdir, subject_id, session_id, "preproc", "transforms", filename)
        
        # Check if file exists
        if os.path.exists(filepath):
            logging.info(f"Found transform {modality} file: {filepath}")
            return filepath
        else:
            logging.debug(f"Transform {modality} file not found at: {filepath}")
            return None

    def _provide_diagnostic_guidance(self, subject_id: str, session_id: str, preproc_outdir: str):
        """Provide diagnostic guidance when files are not found.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            preproc_outdir: Preprocessing output directory
        """
        expected_dir = os.path.join(preproc_outdir, subject_id, session_id, "preproc")
        
        logging.error(f"Expected files not found for {subject_id}_{session_id}")
        logging.error(f"Expected directory: {expected_dir}")
        
        if os.path.exists(expected_dir):
            files = os.listdir(expected_dir)
            logging.error(f"Available files in directory: {files}")
            
            # Check for common naming variations
            t1_variants = [f for f in files if "t1" in f.lower() and "brain" in f.lower()]
            t2_variants = [f for f in files if "t2" in f.lower() and "brain" in f.lower()]
            
            if t1_variants:
                logging.error(f"Found T1 variants: {t1_variants}")
            if t2_variants:
                logging.error(f"Found T2/FLAIR variants: {t2_variants}")
                
        else:
            logging.error(f"Directory does not exist: {expected_dir}")
            # Check if parent directories exist
            parent_dir = os.path.join(preproc_outdir, subject_id, session_id)
            if os.path.exists(parent_dir):
                subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                logging.error(f"Available subdirectories in {parent_dir}: {subdirs}")
            else:
                subject_dir = os.path.join(preproc_outdir, subject_id)
                if os.path.exists(subject_dir):
                    sessions = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
                    logging.error(f"Available sessions for {subject_id}: {sessions}")
                else:
                    subjects = [d for d in os.listdir(preproc_outdir) if os.path.isdir(os.path.join(preproc_outdir, d))]
                    logging.error(f"Available subjects in preprocessing directory: {subjects}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="deepFCD", description="deepFCD model", epilog="I dare you to at the code!"
    )
    parser.add_argument("-bp", "--bidspath", required=True, help="Path to BIDS dataset")
    parser.add_argument(
        "-op",
        "--outpath",
        default="",
        help="Output path (default: <bids_root>/derivatives/deepFCD)",
    )
    parser.add_argument("-sp", "--space", help="Space template")
    parser.add_argument(
        "-bm",
        "--brainmask",
        action="store_true",
        default=False,
        help="Set to True for brain extraction or skull-removal",
    )
    parser.add_argument(
        "-pp",
        "--preprocess",
        action="store_true",
        default=False,
        help="Co-register T1 and T2 images to MNI152 space and N3 correction before brain extraction",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing prediction outputs",
    )
    parser.add_argument(
        "--overwrite-preprocessing",
        action="store_true",
        default=False,
        help="Overwrite existing preprocessing outputs",
    )
    parser.add_argument(
        "-dev", "--device", default="cpu", help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "-s",
        "--subjects",
        nargs="+",
        default=None,
        help="List of subjects to process (default: all). Can include session info like sub-001_ses-01",
    )

    return parser


def main():
    """Main entry point for the deepFCD inference CLI."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Initialize main inference processor
        inference = DeepFCDInference(args)

        # Get subjects from original dataset
        subjects = inference.get_subjects(inference.orig_ds)

        # Initialize and run preprocessing
        preprocessor = DeepFCDPreprocessor(inference)
        preprocessor.preprocess_subjects(subjects)

        # Initialize model handler
        model_handler = DeepFCDModel(inference)
        model_handler.setup_environment()
        model_handler.configure_model_options()
        model_handler.load_trained_model()

        # Initialize processor and run inference
        processor = DeepFCDProcessor(inference, model_handler)
        processor.process_subjects(subjects)

        logging.info("DeepFCD inference completed successfully!")

    except Exception as e:
        logging.error(f"Error during DeepFCD inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
