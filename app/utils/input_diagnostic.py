#!/usr/bin/env python3
"""
Input diagnostic and metadata generation for deepFCD pipeline.
Helps users understand missing files and provides guidance for resolution.
"""

import os
import json
from typing import Dict, List, Optional
from bids import BIDSLayout


class DeepFCDInputDiagnostic:
    """Diagnostic tool for deepFCD input requirements and missing files."""
    
    def __init__(self, bids_path: str, derivatives_path: Optional[str] = None):
        """Initialize the diagnostic tool.
        
        Args:
            bids_path: Path to the BIDS dataset
            derivatives_path: Path to derivatives (if different from bids_path/derivatives)
        """
        self.bids_path = bids_path
        self.derivatives_path = derivatives_path or os.path.join(bids_path, 'derivatives')
        self.orig_layout = None
        self.proc_layout = None
        
        try:
            self.orig_layout = BIDSLayout(bids_path, validate=False)
        except Exception as e:
            print(f"Warning: Could not load original BIDS layout: {e}")
            
        # Check for preprocessing derivatives
        preproc_path = os.path.join(self.derivatives_path, 'deepFCD-preproc')
        if os.path.exists(preproc_path):
            try:
                self.proc_layout = BIDSLayout(preproc_path, validate=False)
            except Exception as e:
                print(f"Warning: Could not load preprocessing derivatives layout: {e}")

    def diagnose_subject(self, subject_id: str, session_id: Optional[str] = None) -> Dict:
        """Diagnose input availability for a specific subject/session.
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-PX034')
            session_id: Session identifier (e.g., 'ses-02') or None
            
        Returns:
            Dictionary with diagnostic information
        """
        if session_id:
            fullid = f"{subject_id}_{session_id}"
        else:
            fullid = subject_id
            
        diagnosis = {
            "subject_id": subject_id,
            "session_id": session_id,
            "fullid": fullid,
            "status": "unknown",
            "missing_files": [],
            "available_files": [],
            "recommendations": [],
            "file_locations": {}
        }
        
        # Check for required preprocessed files
        required_patterns = [
            f"{fullid}_space-MNI152_T1w_brain.nii.gz",
            f"{fullid}_space-MNI152_FLAIR_brain.nii.gz"
        ]
        
        # Look in expected preprocessing output directory
        if '_ses-' in fullid:
            subject_part, session_part = fullid.split('_ses-', 1)
            expected_preproc_dir = os.path.join(
                self.derivatives_path, 'deepFCD-preproc', 
                subject_part, f'ses-{session_part}', 'preproc'
            )
        else:
            expected_preproc_dir = os.path.join(
                self.derivatives_path, 'deepFCD-preproc', 
                fullid, 'preproc'
            )
            
        diagnosis["expected_preproc_dir"] = expected_preproc_dir
        
        # Check for preprocessed files
        found_preprocessed = []
        missing_preprocessed = []
        
        for pattern in required_patterns:
            expected_path = os.path.join(expected_preproc_dir, pattern)
            if os.path.exists(expected_path):
                found_preprocessed.append(expected_path)
                diagnosis["available_files"].append(expected_path)
            else:
                missing_preprocessed.append(pattern)
                diagnosis["missing_files"].append(expected_path)
        
        # Check for original raw files
        raw_files_status = self._check_raw_files(subject_id, session_id)
        diagnosis.update(raw_files_status)
        
        # Determine overall status and recommendations
        if len(found_preprocessed) == len(required_patterns):
            diagnosis["status"] = "ready"
            diagnosis["recommendations"].append("All required preprocessed files found. Ready for inference.")
        elif len(missing_preprocessed) == len(required_patterns):
            diagnosis["status"] = "needs_preprocessing"
            if diagnosis["has_raw_t1"] and diagnosis["has_raw_flair"]:
                diagnosis["recommendations"].extend([
                    "No preprocessed files found, but raw T1w and FLAIR images are available.",
                    "Run preprocessing with: python app/inference_bids.py -bp <path> -pp -bm",
                    "   The -pp flag will automatically generate the required preprocessed files."
                ])
            else:
                diagnosis["recommendations"].extend([
                    "No preprocessed files found and missing raw images.",
                    "Check that your BIDS dataset contains T1w and FLAIR images."
                ])
        else:
            diagnosis["status"] = "partial"
            diagnosis["recommendations"].extend([
                f"Only {len(found_preprocessed)}/{len(required_patterns)} required files found.",
                "Re-run preprocessing to generate missing files."
            ])
            
        return diagnosis
    
    def _check_raw_files(self, subject_id: str, session_id: Optional[str]) -> Dict:
        """Check for raw T1w and FLAIR files in the original dataset."""
        result = {
            "has_raw_t1": False,
            "has_raw_flair": False,
            "raw_t1_path": None,
            "raw_flair_path": None
        }
        
        if not self.orig_layout:
            return result
            
        # Build query for BIDS layout
        query = {"subject": subject_id.replace('sub-', ''), "extension": [".nii.gz", ".nii"]}
        if session_id:
            query["session"] = session_id.replace('ses-', '')
            
        # Check for T1w
        t1_files = self.orig_layout.get(suffix="T1w", **query)
        if t1_files:
            result["has_raw_t1"] = True
            result["raw_t1_path"] = t1_files[0].path
            
        # Check for FLAIR  
        flair_files = self.orig_layout.get(suffix="FLAIR", **query)
        if flair_files:
            result["has_raw_flair"] = True
            result["raw_flair_path"] = flair_files[0].path
            
        return result
    
    def generate_input_specification(self, output_path: str) -> None:
        """Generate a comprehensive input specification file.
        
        Args:
            output_path: Path where to save the input specification
        """
        spec = {
            "name": "deepFCD Input Requirements",
            "description": "Comprehensive specification of input files required by deepFCD",
            "bids_version": "1.7.0",
            "generated_by": "DeepFCDInputDiagnostic",
            "required_preprocessed_files": {
                "description": "deepFCD requires these specific preprocessed files for inference",
                "pattern": "*_space-MNI152_T1w_brain_final.nii.gz and *_space-MNI152_FLAIR_brain_final.nii.gz",
                "location": "derivatives/deepFCD-preproc/sub-<subject>/[ses-<session>/]preproc/",
                "files": [
                    {
                        "suffix": "space-MNI152_T1w_brain_final.nii.gz",
                        "description": "Brain-extracted, bias-corrected, normalized T1w image",
                        "modality": "T1w",
                        "space": "MNI152NLin2009aSym"
                    },
                    {
                        "suffix": "space-MNI152_FLAIR_brain_final.nii.gz", 
                        "description": "Brain-extracted, bias-corrected, normalized FLAIR image",
                        "modality": "FLAIR",
                        "space": "MNI152NLin2009aSym"
                    }
                ]
            },
            "raw_input_requirements": {
                "description": "If preprocessed files are missing, deepFCD can process these raw files",
                "t1w": {
                    "suffix": "T1w", 
                    "datatype": "anat",
                    "extensions": [".nii.gz", ".nii"],
                    "required": True
                },
                "flair": {
                    "suffix": "FLAIR",
                    "datatype": "anat", 
                    "extensions": [".nii.gz", ".nii"],
                    "required": True
                }
            },
            "preprocessing_command": {
                "description": "Command to generate required preprocessed files from raw inputs",
                "command": "python app/inference_bids.py -bp <bids_path> -sp MNI152NLin2009aSym -pp -bm",
                "flags": {
                    "-bp": "Path to BIDS dataset",
                    "-sp": "Spatial reference (MNI152NLin2009aSym recommended)",
                    "-pp": "Enable preprocessing to generate required files",
                    "-bm": "Enable brain masking"
                }
            },
            "troubleshooting": {
                "no_processed_files": {
                    "error_pattern": "No processed T1w files found for sub-*",
                    "cause": "Missing preprocessed files (*_space-MNI152_T1w_brain_final.nii.gz, *_space-MNI152_FLAIR_brain_final.nii.gz)",
                    "solutions": [
                        "Run with -pp flag to enable preprocessing",
                        "Check that raw T1w and FLAIR images exist in BIDS dataset",
                        "Verify BIDS dataset structure and naming conventions"
                    ]
                }
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)
            
        print(f"✓ Generated input specification: {output_path}")

    def scan_dataset(self) -> Dict:
        """Scan the entire dataset and provide a comprehensive report."""
        if not self.orig_layout:
            return {"error": "Could not load BIDS layout"}
            
        subjects = self.orig_layout.get_subjects()
        report = {
            "dataset_path": self.bids_path,
            "total_subjects": len(subjects),
            "subjects_ready": 0,
            "subjects_need_preprocessing": 0,
            "subjects_missing_data": 0,
            "subjects_partial": 0,
            "detailed_results": {}
        }
        
        for subject in subjects:
            subject_id = f"sub-{subject}"
            sessions = self.orig_layout.get_sessions(subject=subject)
            
            if sessions:
                for session in sessions:
                    session_id = f"ses-{session}"
                    diagnosis = self.diagnose_subject(subject_id, session_id)
                    fullid = f"{subject_id}_{session_id}"
                    report["detailed_results"][fullid] = diagnosis
                    
                    if diagnosis["status"] == "ready":
                        report["subjects_ready"] += 1
                    elif diagnosis["status"] == "needs_preprocessing":
                        report["subjects_need_preprocessing"] += 1
                    elif diagnosis["status"] == "partial":
                        report["subjects_partial"] += 1
                    else:
                        report["subjects_missing_data"] += 1
            else:
                diagnosis = self.diagnose_subject(subject_id)
                report["detailed_results"][subject_id] = diagnosis
                
                if diagnosis["status"] == "ready":
                    report["subjects_ready"] += 1
                elif diagnosis["status"] == "needs_preprocessing":
                    report["subjects_need_preprocessing"] += 1
                elif diagnosis["status"] == "partial":
                    report["subjects_partial"] += 1
                else:
                    report["subjects_missing_data"] += 1
                    
        return report
    
    def print_diagnostic_report(self, subjects: List[str] = None) -> None:
        """Print a comprehensive diagnostic report."""
        print("\n" + "="*70)
        print("🔍 deepFCD Input Diagnostic Report")
        print("="*70)
        
        if subjects:
            # Diagnose specific subjects
            for subj in subjects:
                if '_ses-' in subj:
                    subject_id, session_id = subj.split('_ses-', 1)
                    session_id = f"ses-{session_id}"
                else:
                    subject_id, session_id = subj, None
                    
                diagnosis = self.diagnose_subject(subject_id, session_id)
                self._print_subject_diagnosis(diagnosis)
        else:
            # Scan entire dataset
            report = self.scan_dataset()
            self._print_dataset_summary(report)
            
            # Show details for problematic subjects
            for fullid, diagnosis in report["detailed_results"].items():
                if diagnosis["status"] != "ready":
                    self._print_subject_diagnosis(diagnosis)
    
    def _print_subject_diagnosis(self, diagnosis: Dict) -> None:
        """Print diagnosis for a single subject."""
        status_icons = {
            "ready": "✅",
            "needs_preprocessing": "🔧", 
            "partial": "⚠️",
            "unknown": "❓"
        }
        
        icon = status_icons.get(diagnosis["status"], "❓")
        print(f"\n{icon} {diagnosis['fullid']} - {diagnosis['status'].upper()}")
        print("-" * 50)
        
        if diagnosis["available_files"]:
            print("📁 Available files:")
            for f in diagnosis["available_files"]:
                print(f"   ✓ {f}")
                
        if diagnosis["missing_files"]:
            print("❌ Missing files:")
            for f in diagnosis["missing_files"]:
                print(f"   ✗ {f}")
                
        print("💡 Recommendations:")
        for rec in diagnosis["recommendations"]:
            print(f"   {rec}")
    
    def _print_dataset_summary(self, report: Dict) -> None:
        """Print dataset summary."""
        total = report["total_subjects"]
        print(f"\n📊 Dataset Summary ({report['dataset_path']})")
        print(f"   Total subjects/sessions: {total}")
        print(f"   ✅ Ready for inference: {report['subjects_ready']}")
        print(f"   🔧 Need preprocessing: {report['subjects_need_preprocessing']}")
        print(f"   ⚠️  Partial files: {report['subjects_partial']}")
        print(f"   ❌ Missing data: {report['subjects_missing_data']}")


def main():
    """Command-line interface for the diagnostic tool."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose deepFCD input requirements and missing files"
    )
    parser.add_argument(
        "-bp", "--bids_path", 
        required=True,
        help="Path to BIDS dataset"
    )
    parser.add_argument(
        "-s", "--subjects",
        nargs="*",
        help="Specific subjects to diagnose (e.g., sub-PX034_ses-02)"
    )
    parser.add_argument(
        "--generate-spec",
        action="store_true",
        help="Generate input specification file"
    )
    parser.add_argument(
        "--spec-output",
        default="deepFCD_input_specification.json",
        help="Output path for input specification"
    )
    
    args = parser.parse_args()
    
    # Initialize diagnostic tool
    diagnostic = DeepFCDInputDiagnostic(args.bids_path)
    
    # Generate specification if requested
    if args.generate_spec:
        diagnostic.generate_input_specification(args.spec_output)
    
    # Print diagnostic report
    diagnostic.print_diagnostic_report(args.subjects)


if __name__ == "__main__":
    main()
