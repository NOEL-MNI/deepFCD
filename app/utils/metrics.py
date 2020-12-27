import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, jaccard_similarity_score as jc
from nilearn.regions import connected_label_regions

# from numba import autojit, jit, double, vectorize

# @vectorize(['f8(f8,f8)','f4(f4,f4)'])
def dc(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    dc = 2. * intersection.sum() / im_sum

    return dc


def deltaVol(im1, im2):
    """
    absolute difference in volume
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.abs(im2.sum() - im1.sum()) / im1.sum()


def perf_measure_vox(y_pred, y_true):
    # TP = np.zeros(1, dtype=float)
    # FP = np.zeros(1, dtype=float)
    # TN = np.zeros(1, dtype=float)
    # FN = np.zeros(1, dtype=float)

    # for k in range(len(y_pred)):
    #     if y_true[k] == y_pred[k] == 1 :
    #        TP += 1
    #     if y_pred[k] == 1 and y_true[k] != y_pred[k]:
    #        FP += 1
    #     if y_true[k] == y_pred[k] == 0:
    #        TN += 1
    #     if y_pred[k] == 0 and y_true[k] != y_pred[k]:
    #        FN += 1

    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    sensitivity = 100*TP/(TP+FN)
    specificity = 100*TN/(TN+FP)

    print('-'*60)
    print("sensitivity: %.2f" %(sensitivity))
    print("specificity: %.2f" %(specificity))
    print('-'*60)

    perf = {
			'sensitivity': sensitivity,
			'specificity': specificity,
			'TP': TP,
			'FP': FP,
			'TN': TN,
			'FN': FN,
	}

    return perf

# @vectorize(['f8(f8,f8)','f4(f4,f4)'])
def hd(input2, input1):
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    input1 : prediction, array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2 : label, array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Presumably does not influence the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```input1``` and the
        object(s) in ```input2```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric.
    """
    nii2 = nib.Nifti1Image(input2, np.eye(4))
    region_labels = connected_label_regions(nii2)

    conn2 = region_labels.get_data()
    np.unique(conn2)

    conn2	= conn2.astype(np.bool)
    input1	= input1.astype(np.bool)

    hd1 = surf_distances(input1, conn2).max()
    hd2 = surf_distances(conn2, input1).max()
    hd = max(hd1, hd2)

    return -np.log(hd)

# @vectorize(['f8(f8,f8)','f4(f4,f4)'])
def surf_distances(in1, in2):
    """
    The distances between the surface voxel of binary objects in input1 and their
    nearest partner surface voxel of a binary object in input2.
    """
    voxelspacing=None
    connectivity=1
    input1 = np.atleast_1d(in1)
    input2 = np.atleast_1d(in2)
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, input1.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(input1.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(input1):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(input2):
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # extract only 1-pixel border line of objects
    input1_border = input1 - binary_erosion(input1, structure=footprint, iterations=1)
    input2_border = input2 - binary_erosion(input2, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~input2_border, sampling=voxelspacing)
    sds = dt[input1_border]

    return sds
