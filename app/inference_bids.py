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
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils.env import set_theano_flags

try:
    from utils.input_diagnostic import DeepFCDInputDiagnostic
except ImportError:
    print("Warning: Could not import input diagnostic utilities")
    DeepFCDInputDiagnostic = None

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

        # Initialize basic options needed for preprocessing checks
        self._initialize_basic_options()

    def _setup_logging(self):
        """Configure logging settings."""
        # Allow runtime control of logging level via environment variable
        # (DEEPFCD_LOGLEVEL) or the command-line debug flag (args.debug).
        # Default to INFO to avoid verbose debug output.
        env_level = os.environ.get("DEEPFCD_LOGLEVEL")
        if hasattr(self.args, "debug") and getattr(self.args, "debug"):
            level = logging.DEBUG
        else:
            if env_level:
                try:
                    level = getattr(logging, env_level.upper())
                except Exception:
                    level = logging.INFO
            else:
                level = logging.INFO

        logging.basicConfig(
            level=level,
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
            format="{asctime} {levelname} {filename}:{lineno}: {message}",
        )

    def _setup_paths(self):
        """Set up input and output paths."""
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
        self.orig_ds = BIDSLayout(self.args.bidspath, validate=False)

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
            # Parse subjects from subject-session specifications
            subject_session_specs = self.parse_subject_session_specs()
            subjects = list(subject_session_specs.keys())
        return subjects

    def parse_subject_session_specs(self) -> Dict[str, List[str]]:
        """Parse subject-session specifications from command line arguments.

        Returns:
            Dictionary mapping subject IDs to list of requested session IDs (or [None] for all sessions)
        """
        if self.args.subjects is None:
            return {}

        subject_session_specs = {}
        for s in self.args.subjects:
            # Remove 'sub-' prefix if present
            s_clean = s.replace("sub-", "")
            if "_ses-" in s_clean:
                subject_id, session_id = s_clean.split("_ses-", 1)
                if subject_id not in subject_session_specs:
                    subject_session_specs[subject_id] = []
                subject_session_specs[subject_id].append(session_id)
            else:
                # No session specified, include all sessions for this subject
                subject_session_specs[s_clean] = None

        return subject_session_specs

    def get_subject_sessions(self, dataset: BIDSLayout) -> Dict[str, List[str]]:
        """Get sessions for each subject.

        Args:
            dataset: BIDS dataset layout

        Returns:
            Dictionary mapping subject IDs to list of session IDs
        """
        # Get subject-session specifications from command line
        subject_session_specs = self.parse_subject_session_specs()

        if subject_session_specs:
            # Use specified subjects and sessions
            subject_sessions = {}
            for subject_id, requested_sessions in subject_session_specs.items():
                if requested_sessions is None:
                    # Get all sessions for this subject
                    raw_sessions = dataset.get_sessions(subject=subject_id)
                    norm_sessions = []
                    if raw_sessions:
                        for s in raw_sessions:
                            if s is None:
                                norm_sessions.append(None)
                                continue
                            try:
                                s_str = s if isinstance(s, str) else str(s)
                            except Exception:
                                norm_sessions.append(None)
                                continue
                            s_clean = s_str.strip()
                            if s_clean == "" or s_clean.lower() in ("none", "null"):
                                norm_sessions.append(None)
                            else:
                                if s_clean.startswith("ses-"):
                                    s_clean = s_clean.replace("ses-", "")
                                norm_sessions.append(s_clean)
                        subject_sessions[subject_id] = norm_sessions
                    else:
                        subject_sessions[subject_id] = [None]
                else:
                    # Use only requested sessions
                    subject_sessions[subject_id] = requested_sessions
            return subject_sessions
        else:
            # Original logic for when no specific subjects are requested
            subjects = self.get_subjects(dataset)
            subject_sessions = {}

            for subject in subjects:
                raw_sessions = dataset.get_sessions(subject=subject)
                norm_sessions = []
                if raw_sessions:
                    for s in raw_sessions:
                        # Convert pybids NullType or other non-string values to None
                        if s is None:
                            norm_sessions.append(None)
                            continue
                        try:
                            s_str = s if isinstance(s, str) else str(s)
                        except Exception:
                            norm_sessions.append(None)
                            continue
                        s_clean = s_str.strip()
                        if s_clean == "" or s_clean.lower() in ("none", "null"):
                            norm_sessions.append(None)
                        else:
                            # strip leading 'ses-' if present, keep the bare session id
                            if s_clean.startswith("ses-"):
                                s_clean = s_clean.replace("ses-", "")
                            norm_sessions.append(s_clean)
                    subject_sessions[subject] = norm_sessions
                else:
                    # If no sessions, use None to indicate no session structure
                    subject_sessions[subject] = [None]

            return subject_sessions

    def _initialize_basic_options(self):
        """Initialize basic options needed before model setup."""
        # Import options from config
        from config.experiment import options

        # Set basic options needed for preprocessing checks
        options["MNI152space"] = "MNI152"
        options["deepFCD_label"] = "deepFCD"


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
        # Construct expected output directory - BIDS-compliant anat directory
        if "_ses-" in fullid:
            subject_part, session_part = fullid.split("_ses-", 1)
            anat_dir = os.path.join(
                self.inference.preproc_outdir,
                subject_part,
                f"ses-{session_part}",
                "anat",
            )
        else:
            anat_dir = os.path.join(self.inference.preproc_outdir, fullid, "anat")

        logging.debug(f"Checking preprocessing outputs for {fullid} in {anat_dir}")

        # Expected output files - try BIDS-compliant anat directory first, then legacy preproc
        expected_files_anat = [
            os.path.join(
                anat_dir, f"{fullid}_space-{options['MNI152space']}_T1w_brain.nii.gz"
            ),
            os.path.join(
                anat_dir, f"{fullid}_space-{options['MNI152space']}_FLAIR_brain.nii.gz"
            ),
        ]

        # Also check for brain extracted versions and legacy preproc directory
        alternative_files = [
            os.path.join(
                anat_dir, f"{fullid}_space-{options['MNI152space']}_T1w.nii.gz"
            ),
            os.path.join(
                anat_dir, f"{fullid}_space-{options['MNI152space']}_FLAIR.nii.gz"
            ),
        ]

        logging.debug(f"Expected files: {expected_files_anat + alternative_files}")

        # Check if all files exist (primary or alternative)
        files_found = []
        # Check T1w files (try anat first, then alternatives)
        t1_found = False
        for t1_file in [expected_files_anat[0]] + [
            alternative_files[0],  # T1w
        ]:
            if os.path.isfile(t1_file):
                files_found.append(t1_file)
                t1_found = True
                break

        # Check FLAIR files (try anat first, then alternatives)
        flair_found = False
        for flair_file in [expected_files_anat[1]] + [
            alternative_files[1],  # FLAIR
        ]:
            if os.path.isfile(flair_file):
                files_found.append(flair_file)
                flair_found = True
                break

        all_exist = t1_found and flair_found

        if all_exist:
            logging.info(f"Preprocessing outputs already exist for {fullid}")
            for f in files_found:
                logging.debug(f"  Found: {f}")
        else:
            all_candidate_files = expected_files_anat + alternative_files
            missing_files = [f for f in all_candidate_files if not os.path.isfile(f)]
            logging.debug(
                f"Missing preprocessing outputs for {fullid}: {len(missing_files)} files"
            )
            for f in missing_files:
                logging.debug(f"  Missing: {f}")

        return all_exist

    def _provide_diagnostic_guidance(self, fullid: str, missing_modality: str) -> None:
        """Provide detailed guidance when required files are not found.

        Args:
            fullid: Full subject ID (e.g., sub-001_ses-01)
            missing_modality: The modality that's missing (T1w or FLAIR)
        """
        print("\n" + "=" * 70)
        print(f"DeepFCD Input Diagnostics for {fullid}")
        print("=" * 70)

        # Parse subject and session
        if "_ses-" in fullid:
            subject_id, session_part = fullid.split("_ses-", 1)
            session_id = f"ses-{session_part}"
        else:
            subject_id, session_id = fullid, None

        # Expected file patterns
        expected_t1 = f"{fullid}_space-{options['MNI152space']}_T1w_brain.nii.gz"
        expected_t2 = f"{fullid}_space-{options['MNI152space']}_FLAIR_brain.nii.gz"
        expected_t1_alt = f"{fullid}_space-{options['MNI152space']}_T1w.nii.gz"
        expected_t2_alt = f"{fullid}_space-{options['MNI152space']}_FLAIR.nii.gz"

        # Expected directory
        if session_id:
            expected_dir = os.path.join(
                self.inference.preproc_outdir, subject_id, session_id, "anat"
            )
        else:
            expected_dir = os.path.join(self.inference.preproc_outdir, fullid, "anat")

        print(f"MISSING: {missing_modality} preprocessed files for {fullid}")
        print("Expected location: {expected_dir}")
        print("Expected files:")
        print(f"   - {expected_t1}")
        print(f"   - {expected_t2}")
        print("   Or without _brain suffix:")
        print(f"   - {expected_t1_alt}")
        print(f"   - {expected_t2_alt}")

        # Check what's actually available
        print("\nChecking available files...")

        # Check if directory exists
        if os.path.exists(expected_dir):
            files_in_dir = os.listdir(expected_dir)
            if files_in_dir:
                print(f"Directory exists with {len(files_in_dir)} files:")
                for f in sorted(files_in_dir):
                    if f.endswith(".nii.gz"):
                        print(f"   - {f}")
            else:
                print("Directory exists but is empty")
        else:
            print(f"Expected directory does not exist: {expected_dir}")

        # Check for raw files in original dataset
        self._check_raw_files_availability(subject_id, session_id)

        # Provide recommendations
        print("\nRECOMMENDATIONS:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Check if raw files are available
        raw_available = self._check_if_raw_files_exist(subject_id, session_id)

        if raw_available["has_t1"] and raw_available["has_flair"]:
            print("SOLUTION: Raw T1w and FLAIR images found. Run preprocessing:")
            print("")
            print("   python app/inference_bids.py \\")
            print(f"       -bp {self.inference.args.bids_path} \\")
            print(f"       -sp {options['MNI152space']} \\")
            print("       -pp \\")
            print("       -bm")
            print("")
            print("   The -pp flag will automatically generate the required files.")

        elif raw_available["has_t1"] or raw_available["has_flair"]:
            print("PARTIAL DATA: Only some raw images found")
            missing_modalities = []
            if not raw_available["has_t1"]:
                missing_modalities.append("T1w")
            if not raw_available["has_flair"]:
                missing_modalities.append("FLAIR")

            print(f"   Missing: {', '.join(missing_modalities)}")
            print("   Check your BIDS dataset structure")

        else:
            print("NO RAW DATA: No T1w or FLAIR images found in original dataset")
            print("   Required BIDS structure:")
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

        print("\nFor more information:")
        print("   - Input requirements: docs/input_requirements.md")
        print("   - BIDS specification: https://bids-specification.readthedocs.io/")
        print("=" * 70 + "\n")

    def _check_raw_files_availability(
        self, subject_id: str, session_id: Optional[str]
    ) -> None:
        """Check and report on raw file availability."""
        if not self.inference.orig_ds:
            print("⚠️  Cannot check raw files (original dataset not loaded)")
            return

        # Build query
        query = {
            "subject": subject_id.replace("sub-", ""),
            "extension": [".nii.gz", ".nii"],
        }
        if session_id:
            query["session"] = session_id.replace("ses-", "")

        # Check T1w
        t1_files = self.inference.orig_ds.get(suffix="T1w", **query)
        flair_files = self.inference.orig_ds.get(suffix="FLAIR", **query)

        print("Raw file availability:")
        if t1_files:
            print(f"   T1w: {t1_files[0].path}")
        else:
            print("   T1w: Not found")

        if flair_files:
            print(f"   FLAIR: {flair_files[0].path}")
        else:
            print("   FLAIR: Not found")

    def _check_if_raw_files_exist(
        self, subject_id: str, session_id: Optional[str]
    ) -> Dict[str, bool]:
        """Check if raw files exist and return status."""
        result = {"has_t1": False, "has_flair": False}

        if not self.inference.orig_ds:
            return result

        # Build query
        query = {
            "subject": subject_id.replace("sub-", ""),
            "extension": [".nii.gz", ".nii"],
        }
        if session_id:
            query["session"] = session_id.replace("ses-", "")

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

        # Import preprocess_image only when brain masking and pre-processing is enabled
        try:
            from preprocess_bids import preprocess_image
        except ImportError:
            logging.error(
                "preprocess_image function is not available. Cannot perform preprocessing."
            )
            raise ImportError(
                "preprocess_image function could not be imported from preprocess_bids"
            )

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
                    t1w_files = self.inference.orig_ds.get(**query_params)
                    if not t1w_files:
                        raise IndexError("No T1w files found")
                    t1w_file = t1w_files[0]

                    # Update query for FLAIR
                    query_params["suffix"] = "FLAIR"
                    flair_files = self.inference.orig_ds.get(**query_params)
                    if not flair_files:
                        raise IndexError("No FLAIR files found")
                    flair_file = flair_files[0]

                    # Create appropriate full ID
                    if session is not None:
                        fullid = f"sub-{subject}_ses-{session}"
                    else:
                        fullid = f"sub-{subject}"

                    # Check if preprocessing outputs already exist
                    if self._check_preprocessing_outputs_exist(fullid):
                        if self.inference.args.overwrite_pp:
                            logging.info(
                                f"Preprocessing outputs exist for {fullid}, but --overwrite-pp specified"
                            )
                        else:
                            logging.info(
                                f"Skipping preprocessing for {fullid} (outputs already exist, use --overwrite-pp to force)"
                            )
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
                logging.info(f"  {i + 1}. {fullid}")

        if fullids:
            logging.info("Will process:")
            for i, fullid in enumerate(fullids):
                logging.info(f"  {i + 1}. {fullid}")

        if not fullids:
            if skipped_subjects:
                logging.info(
                    "All preprocessing outputs already exist. Use --overwrite to force reprocessing."
                )
            else:
                logging.warning(
                    "No valid subject-session combinations found for preprocessing"
                )
            return

        # Process images in parallel using the preprocessing derivatives directory
        # Use max_workers=1 to avoid confusion with multiple parallel processes
        process_map(
            partial(
                preprocess_image,
                indir_=self.inference.args.bidspath,
                outdir_=self.inference.preproc_outdir,
                preprocess=self.inference.args.preprocess,
                use_gpu=0,  # prefer CPU for parallel preprocessing
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

        # Validate that critical functions were imported successfully
        if test_model is None:
            raise ImportError("Failed to import test_model function from utils.base")
        if transform_img is None:
            raise ImportError("Failed to import transform_img function from utils.base")

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

        # Validate critical paths
        if options["test_folder"] is None:
            raise ValueError("test_folder is None - check outdir configuration")
        if not isinstance(options["test_folder"], (str, bytes, os.PathLike)):
            raise ValueError(
                f"test_folder is not a valid path type: {type(options['test_folder'])} = {options['test_folder']}"
            )
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

    def _normalize_session(self, session_id: Optional[str]) -> Optional[str]:
        """Normalize session identifier values coming from pybids or user input.

        This converts pybids NullType or other non-string/null-like values to
        Python None, and ensures string sessions are returned in the
        'ses-XXX' form when appropriate.
        """
        if session_id is None:
            return None
        # Convert non-string session identifiers (e.g. pybids NullType) to string
        try:
            s = session_id if isinstance(session_id, str) else str(session_id)
        except Exception:
            return None

        s_clean = s.strip()
        if s_clean == "":
            return None
        if s_clean.lower() in ("none", "null"):
            return None
        # If already in bids form, return as-is
        if s_clean.startswith("ses-"):
            return s_clean
        # Otherwise prefix
        return f"ses-{s_clean}"

    def process_subjects(self, subjects: List[str]):
        """Run inference on all subjects.

        Args:
            subjects: List of subject IDs to process
        """
        # Check if preprocessing output directory exists and has content
        if not os.path.exists(self.inference.preproc_outdir):
            logging.error(
                f"Preprocessing output directory does not exist: {self.inference.preproc_outdir}"
            )
            logging.error("Please run preprocessing first with the -bm flag")
            return

        # Check if preprocessing directory has any subject folders
        subject_dirs = [
            d
            for d in os.listdir(self.inference.preproc_outdir)
            if os.path.isdir(os.path.join(self.inference.preproc_outdir, d))
            and d.startswith("sub-")
        ]

        if not subject_dirs:
            logging.error(
                f"No preprocessed subjects found in: {self.inference.preproc_outdir}"
            )
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
            logging.debug(f"test_model function: {test_model}")
            logging.debug(f"transform_img function: {transform_img}")
            if test_model is None:
                logging.error(f"test_model is None for {fullid}")
            if transform_img is None:
                logging.error(f"transform_img is None for {fullid}")
            self._process_single_subject(
                subject_id, session_id, fullid, test_model, transform_img
            )
            logging.info(f"Inference finished for {fullid}")

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
            logging.info(
                f"Skipping {fullid}: required input or transform files not found"
            )
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
            raise ValueError(
                f"test_folder is not a valid path type for {fullid}: {type(options['test_folder'])} = {options['test_folder']}"
            )

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
        try:
            logging.info(f"Created pred_folder: {options['pred_folder']}")
            logging.info(
                f"test_folder exists: {os.path.exists(options['test_folder'])}"
            )
            logging.info(
                f"Contents of test_folder: {os.listdir(options['test_folder'])}"
            )
            logging.info(
                f"Contents of pred_folder: {os.listdir(options['pred_folder'])}"
            )
        except Exception:
            logging.exception("Could not list prediction directories for debugging")

        # Check if predictions already exist
        pred_files = self._get_prediction_file_paths(fullid)
        if self._predictions_exist(
            pred_files, orig_bidsfiles, orig_files, transform_files, transform_img_func
        ):
            logging.info(
                f"Skipping {fullid}: predictions already exist and overwrite not set"
            )
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
                "space": options["MNI152space"],
                "label": "brain",
                "extension": ".nii.gz",
            }

            # Build query parameters for original files
            orig_query = {"subject": subject_id, "extension": ".nii.gz"}

            # Build query parameters for transform files
            transform_query = {"subject": subject_id, "extension": "mat"}

            # Normalize session identifiers to avoid passing non-string values
            norm_session = self._normalize_session(session_id)
            if norm_session is not None:
                # BIDSLayout expects the session value without the 'ses-' prefix
                proc_query["session"] = norm_session.replace("ses-", "")
                orig_query["session"] = norm_session.replace("ses-", "")
                transform_query["session"] = norm_session.replace("ses-", "")

            # Get T1 files - try direct file path first, then BIDS query
            bids_subject_id = (
                f"sub-{subject_id}" if not subject_id.startswith("sub-") else subject_id
            )
            bids_session_id = None
            norm_session = self._normalize_session(session_id)
            if norm_session is not None:
                bids_session_id = norm_session

            t1_file = self._get_preprocessed_file(
                bids_subject_id, bids_session_id, "T1w", self.inference.preproc_outdir
            )
            if not t1_file:
                # Fallback to BIDS query for T1w
                proc_query["suffix"] = "T1w"
                t1_proc_files = self.inference.proc_ds.get(**proc_query)
                if not t1_proc_files:
                    logging.error(f"No processed T1w files found for {fullid}")
                    self._provide_diagnostic_guidance(
                        bids_subject_id, bids_session_id, self.inference.preproc_outdir
                    )
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
            t1_transform = self._get_transform_file(
                bids_subject_id, bids_session_id, "T1w", self.inference.preproc_outdir
            )
            logging.debug(
                f"T1 transform from _get_transform_file: {t1_transform} (type: {type(t1_transform)})"
            )
            if not t1_transform:
                # Fallback to BIDS query for T1w transforms
                transform_query["suffix"] = "T1w"
                t1_transform_files = self.inference.proc_ds.get(**transform_query)
                logging.debug(f"T1 transform BIDS query result: {t1_transform_files}")
                if not t1_transform_files:
                    logging.error(f"No T1w transform files found for {fullid}")
                    return None
                t1_transform = t1_transform_files[0].path
                logging.debug(
                    f"T1 transform from BIDS query: {t1_transform} (type: {type(t1_transform)})"
                )
                if t1_transform is None:
                    logging.error(f"T1w transform path is None for {fullid}")
                    return None

            # Get FLAIR files
            # Get FLAIR/T2 files - try direct file path first, then BIDS query
            t2_file = self._get_preprocessed_file(
                bids_subject_id, bids_session_id, "FLAIR", self.inference.preproc_outdir
            )
            if not t2_file:
                # Fallback to BIDS query for FLAIR
                proc_query["suffix"] = "FLAIR"
                t2_proc_files = self.inference.proc_ds.get(**proc_query)
                if not t2_proc_files:
                    logging.error(f"No processed FLAIR files found for {fullid}")
                    self._provide_diagnostic_guidance(
                        bids_subject_id, bids_session_id, self.inference.preproc_outdir
                    )
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
            t2_transform = self._get_transform_file(
                bids_subject_id, bids_session_id, "FLAIR", self.inference.preproc_outdir
            )
            logging.debug(
                f"FLAIR transform from _get_transform_file: {t2_transform} (type: {type(t2_transform)})"
            )
            if not t2_transform:
                # Fallback to BIDS query for FLAIR transforms
                transform_query["suffix"] = "FLAIR"
                t2_transform_files = self.inference.proc_ds.get(**transform_query)
                logging.debug(
                    f"FLAIR transform BIDS query result: {t2_transform_files}"
                )
                if not t2_transform_files:
                    logging.error(f"No FLAIR transform files found for {fullid}")
                    return None
                t2_transform = t2_transform_files[0].path
                logging.debug(
                    f"FLAIR transform from BIDS query: {t2_transform} (type: {type(t2_transform)})"
                )
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
            logging.error(
                f"  Processed dataset directory: {self.inference.preproc_outdir}"
            )
            logging.error(
                f"  Original dataset directory: {self.inference.args.bidspath}"
            )
            # List available files for debugging
            if hasattr(self.inference, "proc_ds") and self.inference.proc_ds:
                try:
                    available_files = self.inference.proc_ds.get(subject=subject_id)
                    logging.error(
                        f"  Available processed files for sub-{subject_id}: {len(available_files)} files"
                    )
                    for f in available_files[:5]:  # Show first 5 files
                        logging.error(f"    {f.path}")
                    if len(available_files) > 5:
                        logging.error(
                            f"    ... and {len(available_files) - 5} more files"
                        )
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
            raise ValueError(
                f"pred_folder is not a valid path type for {fullid}: {type(options['pred_folder'])} = {options['pred_folder']}"
            )

        # Create BIDS-compliant filenames for predictions
        base_filename = f"{fullid}_space-{options['MNI152space']}_desc-deepFCD"

        pred_mean_fname = os.path.join(
            options["pred_folder"],
            f"{base_filename}_stat-mean1_probseg.nii.gz",
        )
        pred_var_fname = os.path.join(
            options["pred_folder"],
            f"{base_filename}_stat-mean1_probseg.nii.gz",
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
                targetspace = "orig"
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
        logging.debug(f"About to call test_model_func: {test_model_func}")
        logging.info(f"Pred folder for {fullid}: {options.get('pred_folder')}")
        logging.info(
            f"Planned output names: mean={os.path.join(options.get('pred_folder', ''), options.get('fullid', '') + '_space-' + options.get('MNI152space', '') + '_stat-mean1_probseg.nii.gz')}"
        )
        if test_model_func is None:
            logging.error(f"test_model_func is None for {fullid}")
            raise ValueError(f"test_model_func is None for {fullid}")

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

        logging.info(f"test_model returned outputs: {list(outputs.keys())}")

        # Transform outputs back to original space
        for k, v in outputs.items():
            targetspace = "orig"  # Always use "orig" to indicate original/native space

            logging.debug(f"About to call transform_img_func: {transform_img_func}")
            if transform_img_func is None:
                logging.error(f"transform_img_func is None for {fullid}")
                raise ValueError(f"transform_img_func is None for {fullid}")

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

    def _get_preprocessed_file(
        self, subject_id: str, session_id: str, modality: str, preproc_outdir: str
    ) -> Optional[str]:
        """Get preprocessed file path by checking direct file paths.

        Args:
            subject_id: Subject identifier (e.g., 'sub-PX034')
            session_id: Session identifier (e.g., 'ses-02' or None)
            modality: Modality type (T1w or FLAIR)
            preproc_outdir: Preprocessing output directory

        Returns:
            Path to preprocessed file if found, None otherwise
        """
        # Construct the full ID
        # session_id may be like 'ses-01' or None
        if session_id and session_id not in ("None", "null"):
            # remove any leading 'ses-' for filesystem paths when joining
            sess = (
                session_id.replace("ses-", "")
                if session_id.startswith("ses-")
                else session_id
            )
            fullid = f"{subject_id}_ses-{sess}"
            # BIDS-compliant anat directory
            anat_dir = os.path.join(preproc_outdir, subject_id, f"ses-{sess}", "anat")
        else:
            fullid = subject_id
            # BIDS-compliant anat directory
            anat_dir = os.path.join(preproc_outdir, subject_id, "anat")

        # Determine file suffix based on modality - using actual preprocessing output naming
        if modality == "T1w":
            actual_filename = f"{fullid}_space-{options['MNI152space']}_T1w.nii.gz"
            brain_filename = f"{fullid}_space-{options['MNI152space']}_T1w_brain.nii.gz"
        elif modality == "FLAIR":
            actual_filename = f"{fullid}_space-{options['MNI152space']}_FLAIR.nii.gz"
            brain_filename = (
                f"{fullid}_space-{options['MNI152space']}_FLAIR_brain.nii.gz"
            )
        else:
            logging.warning(f"Unknown modality: {modality}")
            return None

        # Try different file paths in order of preference
        candidate_paths = [
            # BIDS-compliant anat directory with brain extracted naming (preferred)
            os.path.join(anat_dir, brain_filename),
            # BIDS-compliant anat directory with actual preprocessing naming
            os.path.join(anat_dir, actual_filename),
        ]

        for filepath in candidate_paths:
            if os.path.exists(filepath):
                logging.info(f"Found preprocessed {modality} file: {filepath}")
                return filepath

        # Log all attempted paths for debugging
        logging.debug(f"Preprocessed {modality} file not found. Tried paths:")
        for path in candidate_paths:
            logging.debug(f"  - {path}")

        return None

    def _get_transform_file(
        self, subject_id: str, session_id: str, modality: str, preproc_outdir: str
    ) -> Optional[str]:
        """Get transform file path by checking direct file paths.

        Args:
            subject_id: Subject identifier (e.g., 'sub-PX034')
            session_id: Session identifier (e.g., 'ses-02' or None)
            modality: Modality type (T1w or FLAIR)
            preproc_outdir: Preprocessing output directory

        Returns:
            Path to transform file if found, None otherwise
        """
        # Construct the full ID
        # session_id may be like 'ses-01' or None
        if session_id and session_id not in ("None", "null"):
            sess = (
                session_id.replace("ses-", "")
                if session_id.startswith("ses-")
                else session_id
            )
            fullid = f"{subject_id}_ses-{sess}"
            # BIDS-compliant xfm directory at session level
            xfm_dir = os.path.join(preproc_outdir, subject_id, f"ses-{sess}", "xfm")
            # BIDS-compliant anat directory
            anat_dir = os.path.join(preproc_outdir, subject_id, f"ses-{sess}", "anat")
        else:
            fullid = subject_id
            # BIDS-compliant xfm directory at subject level
            xfm_dir = os.path.join(preproc_outdir, subject_id, "xfm")
            # BIDS-compliant anat directory
            anat_dir = os.path.join(preproc_outdir, subject_id, "anat")

        # Determine file suffix based on modality - BEP014 compliant naming
        if modality == "T1w":
            bep014_filename = (
                f"{fullid}_from-T1w_to-{options['MNI152space']}_mode-image_xfm.mat"
            )
        elif modality == "FLAIR":
            bep014_filename = (
                f"{fullid}_from-FLAIR_to-{options['MNI152space']}_mode-image_xfm.mat"
            )
        else:
            logging.warning(f"Unknown modality: {modality}")
            return None

        # Try different file paths in order of preference
        candidate_paths = [
            # BIDS-compliant xfm directory with BEP014 naming
            os.path.join(xfm_dir, bep014_filename),
            # BIDS-compliant anat directory with BEP014 naming
            os.path.join(anat_dir, bep014_filename),
        ]

        for filepath in candidate_paths:
            if os.path.exists(filepath):
                logging.info(f"Found transform {modality} file: {filepath}")
                return filepath

        # Log all attempted paths for debugging
        logging.debug(f"Transform {modality} file not found. Tried paths:")
        for path in candidate_paths:
            logging.debug(f"  - {path}")

        return None

    def _provide_diagnostic_guidance(
        self, subject_id: str, session_id: str, preproc_outdir: str
    ):
        """Provide diagnostic guidance when files are not found.

        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            preproc_outdir: Preprocessing output directory
        """
        # session_id may be like 'ses-01' or None
        if session_id and session_id not in ("None", "null"):
            sess = (
                session_id.replace("ses-", "")
                if session_id.startswith("ses-")
                else session_id
            )
            expected_dir = os.path.join(
                preproc_outdir, subject_id, f"ses-{sess}", "anat"
            )
            lookup_key = f"{subject_id}_ses-{sess}"
        else:
            expected_dir = os.path.join(preproc_outdir, subject_id, "anat")
            lookup_key = subject_id

        logging.error(f"Expected files not found for {lookup_key}")
        logging.error(f"Expected directory: {expected_dir}")

        if os.path.exists(expected_dir):
            files = os.listdir(expected_dir)
            logging.error(f"Available files in directory: {files}")

            # Check for common naming variations
            t1_variants = [
                f for f in files if "t1" in f.lower() and "brain" in f.lower()
            ]
            t2_variants = [
                f for f in files if "t2" in f.lower() and "brain" in f.lower()
            ]

            if t1_variants:
                logging.error(f"Found T1 variants: {t1_variants}")
            if t2_variants:
                logging.error(f"Found T2/FLAIR variants: {t2_variants}")
        else:
            logging.error(f"Directory does not exist: {expected_dir}")
            # Check if parent directories exist
            if session_id and session_id not in ("None", "null"):
                sess = (
                    session_id.replace("ses-", "")
                    if session_id.startswith("ses-")
                    else session_id
                )
                parent_dir = os.path.join(preproc_outdir, subject_id, f"ses-{sess}")
            else:
                parent_dir = os.path.join(preproc_outdir, subject_id)

            if os.path.exists(parent_dir):
                subdirs = [
                    d
                    for d in os.listdir(parent_dir)
                    if os.path.isdir(os.path.join(parent_dir, d))
                ]
                logging.error(f"Available subdirectories in {parent_dir}: {subdirs}")
            else:
                subjects = [
                    d
                    for d in os.listdir(preproc_outdir)
                    if os.path.isdir(os.path.join(preproc_outdir, d))
                ]
                logging.error(
                    f"Available subjects in preprocessing directory: {subjects}"
                )


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
        "--overwrite-pp",
        "--overwrite-preproc",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging output",
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

    except Exception:
        # The logging subsystem in the runtime may be misconfigured; print a full traceback
        import traceback

        print("Error during DeepFCD inference. Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
