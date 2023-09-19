"""
Test deepFCD.py

nptest.assert_allclose
self.assertEqual
self.assertTrue

"""

import os
import unittest
from tempfile import mktemp

import ants
import numpy as np
import numpy.testing as nptest

from utils import compare_images


params = {}
if os.environ.get("CI_TESTING") is not None:
    params["CI_TESTING_PRED_DIR"] = os.environ.get("CI_TESTING_PRED_DIR")
    params["CI_TESTING_PATIENT_ID"] = os.environ.get("CI_TESTING_PATIENT_ID")
else:
    params["CI_TESTING_PRED_DIR"] = "/host/hamlet/local_raid/data/ravnoor/sandbox/pytests"
    params["CI_TESTING_PATIENT_ID"] = "sub-00055"


class TestModule_deepFCD(unittest.TestCase):

    def setUp(self):
        # load predictions from a previous validated run (known as ground-truth labels in this context)
        self.gt_deepMask = ants.image_read('segmentations/sub-00055/sub-00055_brain_mask_final.nii.gz').clone('unsigned int')
        self.gt_deepFCD_mean = ants.image_read('segmentations/sub-00055/noel_deepFCD_dropoutMC/sub-00055_noel_deepFCD_dropoutMC_prob_mean_1.nii.gz').clone('float')
        self.gt_deepFCD_var = ants.image_read('segmentations/sub-00055/noel_deepFCD_dropoutMC/sub-00055_noel_deepFCD_dropoutMC_prob_var_1.nii.gz').clone('float')
        
        pred_path = os.path.join(params["CI_TESTING_PRED_DIR"], params["CI_TESTING_PATIENT_ID"])
        # load predicitions from the most recent run
        self.pred_deepMask = ants.image_read(pred_path + '/' + params["CI_TESTING_PATIENT_ID"] + '_brain_mask_final.nii.gz').clone('unsigned int')
        self.pred_deepFCD_mean = ants.image_read(pred_path + '/noel_deepFCD_dropoutMC/' + params["CI_TESTING_PATIENT_ID"] + '_noel_deepFCD_dropoutMC_prob_mean_1.nii.gz').clone('float')
        self.pred_deepFCD_var = ants.image_read(pred_path + '/noel_deepFCD_dropoutMC/' + params["CI_TESTING_PATIENT_ID"] + '_noel_deepFCD_dropoutMC_prob_var_1.nii.gz').clone('float')

        self.imgs = [self.pred_deepMask, self.pred_deepFCD_mean, self.pred_deepFCD_var]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_image_header_info(self):
        # def image_header_info(filename):
        for img in self.imgs:
            img.set_spacing([6.9]*img.dimension)
            img.set_origin([3.6]*img.dimension)
            tmpfile = mktemp(suffix='.nii.gz')
            ants.image_write(img, tmpfile)

            info = ants.image_header_info(tmpfile)
            self.assertEqual(info['dimensions'], img.shape)
            nptest.assert_allclose(info['direction'], img.direction)
            self.assertEqual(info['nComponents'], img.components)
            self.assertEqual(info['nDimensions'], img.dimension)
            self.assertEqual(info['origin'], img.origin)
            self.assertEqual(info['pixeltype'], img.pixeltype)
            self.assertEqual(info['pixelclass'], 'vector' if img.has_components else 'scalar')
            self.assertEqual(info['spacing'], img.spacing)

            try:
                os.remove(tmpfile)
            except:
                pass

        # non-existent file
        with self.assertRaises(Exception):
            tmpfile = mktemp(suffix='.nii.gz')
            ants.image_header_info(tmpfile)


    def test_image_read_write(self):
        # def image_read(filename, dimension=None, pixeltype='float'):
        # def image_write(image, filename):

        # test scalar images
        for img in self.imgs:
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255.
            img = img.clone('unsigned char')
            for ptype in self.pixeltypes:
                img = img.clone(ptype)
                tmpfile = mktemp(suffix='.nii.gz')
                ants.image_write(img, tmpfile)

                img2 = ants.image_read(tmpfile)
                self.assertTrue(ants.image_physical_space_consistency(img,img2))
                self.assertEqual(img2.components, img.components)
                nptest.assert_allclose(img.numpy(), img2.numpy())

            # unsupported ptype
            with self.assertRaises(Exception):
                ants.image_read(tmpfile, pixeltype='not-suppoted-ptype')

        # test saving/loading as npy
        for img in self.imgs:
            tmpfile = mktemp(suffix='.npy')
            ants.image_write(img, tmpfile)
            img2 = ants.image_read(tmpfile)

            self.assertTrue(ants.image_physical_space_consistency(img,img2))
            self.assertEqual(img2.components, img.components)
            nptest.assert_allclose(img.numpy(), img2.numpy())

            # with no json header
            arr = img.numpy()
            tmpfile = mktemp(suffix='.npy')
            np.save(tmpfile, arr)
            img2 = ants.image_read(tmpfile)
            nptest.assert_allclose(img.numpy(), img2.numpy())

        # non-existant file
        with self.assertRaises(Exception):
            tmpfile = mktemp(suffix='.nii.gz')
            ants.image_read(tmpfile)


    def test_brain_mask_segmentation(self):
        metric = compare_images(self.gt_deepMask, self.pred_deepMask)
        print("overlap of the brain mask with the label: {}".format(metric))
        # set relative tolerance to 0.05
        # predicted image is expected to have overlap within 0.05
        nptest.assert_allclose(1., metric, rtol=0.05, atol=0)


    def test_deepFCD_segmentation_mean(self):
        metric = compare_images(self.gt_deepFCD_mean, self.pred_deepFCD_mean, metric_type='correlation')
        print("correlation of the mean probability map with the the label: {}".format(metric))
        # set relative tolerance to 0.05
        # predicted image is expected to have correlation within 0.05
        nptest.assert_allclose(1., metric, rtol=0.05, atol=0)


    def test_deepFCD_segmentation_var(self):
        metric = compare_images(self.gt_deepFCD_var, self.pred_deepFCD_var, metric_type='correlation')
        print("correlation of the mean uncertainty map with the the label: {}".format(metric))
        # set relative tolerance to 0.05
        # predicted image is expected to have correlation within 0.05
        nptest.assert_allclose(1., metric, rtol=0.05, atol=0)


if __name__ == '__main__':
    unittest.main()