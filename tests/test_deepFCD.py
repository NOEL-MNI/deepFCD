"""
Test deepFCD.py
"""

import os
import unittest
from tempfile import mktemp

import ants
import numpy.testing as nptest

from utils import compare_images


class TestDeepFCD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test parameters once for all tests."""
        cls.patient_id = os.environ.get("CI_TESTING_PATIENT_ID", "sub-00055")
        cls.pred_dir = os.environ.get(
            "CI_TESTING_PRED_DIR",
            "/host/hamlet/local_raid/data/ravnoor/sandbox/pytests",
        )
        cls.prediction_path = os.path.join(cls.pred_dir, cls.patient_id)

        # Ground truth files
        cls.gt_files = {
            "brain_mask": f"segmentations/{cls.patient_id}/{cls.patient_id}_brain_mask_final.nii.gz",
            "fcd_mean": f"segmentations/{cls.patient_id}/noel_deepFCD_dropoutMC/{cls.patient_id}_noel_deepFCD_dropoutMC_prob_mean_1.nii.gz",
            "fcd_var": f"segmentations/{cls.patient_id}/noel_deepFCD_dropoutMC/{cls.patient_id}_noel_deepFCD_dropoutMC_prob_var_1.nii.gz",
        }

    def setUp(self):
        """Load ground truth and prediction images."""
        # Load ground truth images
        self.gt_images = {
            "brain_mask": ants.image_read(self.gt_files["brain_mask"]).clone(
                "unsigned int"
            ),
            "fcd_mean": ants.image_read(self.gt_files["fcd_mean"]).clone("float"),
            "fcd_var": ants.image_read(self.gt_files["fcd_var"]).clone("float"),
        }

        # Load prediction images
        self.pred_files = {
            "brain_mask": os.path.join(
                self.prediction_path, f"{self.patient_id}_brain_mask_final.nii.gz"
            ),
            "fcd_mean": os.path.join(
                self.prediction_path,
                "noel_deepFCD_dropoutMC",
                f"{self.patient_id}_noel_deepFCD_dropoutMC_prob_mean_1.nii.gz",
            ),
            "fcd_var": os.path.join(
                self.prediction_path,
                "noel_deepFCD_dropoutMC",
                f"{self.patient_id}_noel_deepFCD_dropoutMC_prob_var_1.nii.gz",
            ),
        }

        self.pred_images = {
            "brain_mask": ants.image_read(self.pred_files["brain_mask"]).clone(
                "unsigned int"
            ),
            "fcd_mean": ants.image_read(self.pred_files["fcd_mean"]).clone("float"),
            "fcd_var": ants.image_read(self.pred_files["fcd_var"]).clone("float"),
        }

    def _test_image_comparison(self, image_type, metric_type="overlap", tolerance=0.05):
        """Helper method to test image comparisons."""
        gt_img = self.gt_images[image_type]
        pred_img = self.pred_images[image_type]

        print(f"Comparing {image_type}:")
        print(f"  Ground truth image: {self.gt_files[image_type]}")
        print(f"  Prediction image: {self.pred_files[image_type]}")
        print(f"  Ground truth shape: {gt_img.shape}")
        print(f"  Prediction shape: {pred_img.shape}")

        metric = compare_images(gt_img, pred_img, metric_type=metric_type)
        print(f"  {metric_type}: {metric}")

        nptest.assert_allclose(1.0, metric, rtol=tolerance, atol=0)

    def test_brain_mask_segmentation(self):
        """Test brain mask segmentation overlap."""
        self._test_image_comparison("brain_mask", "overlap")

    def test_deepFCD_mean_segmentation(self):
        """Test mean probability map correlation."""
        self._test_image_comparison("fcd_mean", "correlation")

    def test_deepFCD_var_segmentation(self):
        """Test variance map correlation."""
        self._test_image_comparison("fcd_var", "correlation")

    def test_image_io_operations(self):
        """Test image read/write operations."""
        test_images = list(self.pred_images.values())
        pixeltypes = ["unsigned char", "unsigned int", "float"]

        for img in test_images:
            # Normalize image for testing
            normalized_img = (img - img.min()) / (img.max() - img.min()) * 255.0
            normalized_img = normalized_img.clone("unsigned char")

            for ptype in pixeltypes:
                test_img = normalized_img.clone(ptype)
                tmpfile = mktemp(suffix=".nii.gz")

                try:
                    ants.image_write(test_img, tmpfile)
                    loaded_img = ants.image_read(tmpfile)

                    self.assertTrue(
                        ants.image_physical_space_consistency(test_img, loaded_img)
                    )
                    self.assertEqual(loaded_img.components, test_img.components)
                    nptest.assert_allclose(test_img.numpy(), loaded_img.numpy())
                finally:
                    if os.path.exists(tmpfile):
                        os.remove(tmpfile)

    def test_image_header_info(self):
        """Test image header information."""
        for img in self.pred_images.values():
            img.set_spacing([6.9] * img.dimension)
            img.set_origin([3.6] * img.dimension)
            tmpfile = mktemp(suffix=".nii.gz")

            try:
                ants.image_write(img, tmpfile)
                info = ants.image_header_info(tmpfile)

                self.assertEqual(info["dimensions"], img.shape)
                nptest.assert_allclose(info["direction"], img.direction)
                self.assertEqual(info["nComponents"], img.components)
                self.assertEqual(info["nDimensions"], img.dimension)
                self.assertEqual(info["origin"], img.origin)
                self.assertEqual(info["pixeltype"], img.pixeltype)
                self.assertEqual(info["spacing"], img.spacing)
            finally:
                if os.path.exists(tmpfile):
                    os.remove(tmpfile)


if __name__ == "__main__":
    unittest.main()
