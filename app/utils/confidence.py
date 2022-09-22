import os

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel import load as load_nii
from nilearn.plotting import find_parcellation_cut_coords
from scipy import ndimage as nd
from sklearn.preprocessing import minmax_scale


def find_center_xyz(labels_np, header):
    labels_img = nib.Nifti1Image(labels_np, affine=header.get_qform(), header=header)
    coords = find_parcellation_cut_coords(
        labels_img, background_label=0, return_label_names=False
    )
    new_coords = [np.round(coord, 2) for coord in coords[0]]
    return new_coords


def get_rank_array(unsorted_array):
    order = unsorted_array.argsort()
    ranks = order.argsort()
    return ranks


def assign_rank_io(
    pid,
    lesion_fp_bin,
    data_prob,
    data_var,
    header,
    options,
    pred_labels,
    label_list,
):

    struct = nd.generate_binary_structure(3, 3)
    pred_labels, _ = nd.label(lesion_fp_bin, structure=struct)
    label_list = np.unique(pred_labels)  # drop the background label

    num_elements_by_lesion = nd.labeled_comprehension(
        lesion_fp_bin, pred_labels, label_list, np.sum, float, 0
    )

    tmp = np.zeros_like(lesion_fp_bin)
    uncert, prob, coords = [], [], []
    #     print(num_elements_by_lesion, type(num_elements_by_lesion))
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > options["l_min"]:
            # assign voxels to output
            current_voxels = np.stack(np.where(pred_labels == l), axis=1)
            tmp[current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]] = 1
            coord = find_center_xyz(tmp, header)
            #             print(coord)
            coords.append(coord)
            uncert.append(np.median(np.ma.masked_equal(tmp * data_var, 0).compressed()))
            prob.append(np.median(np.ma.masked_equal(tmp * data_prob, 0).compressed()))
            tmp = np.zeros_like(lesion_fp_bin)

    #     print(1/np.array(uncert).ravel())
    conf = 100 * minmax_scale(1 / np.array(uncert).ravel())
    conf_sort = 1 + get_rank_array(-conf)  # reverse arg_sort() and offset zero rank
    #     conf_sort
    #     print(conf_sort)
    output_scan = np.zeros_like(lesion_fp_bin)
    # les_lab = []
    for l in range(1, len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > options["l_min"]:
            # assign voxels to output
            current_voxels = np.stack(np.where(pred_labels == l), axis=1)
            output_scan[
                current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]
            ] = conf_sort[l - 1]

    out_img = nib.Nifti1Image(output_scan, affine=header.get_qform(), header=header)
    fname = os.path.join(options["data_folder"], str(pid) + "_ranked_image.nii.gz")
    nib.save(out_img, fname)
    return np.array(prob).ravel(), np.array(conf).ravel(), conf_sort, uncert, coords


def extractLesionCluster(scan, ea, ea_var, options):
    ea_orig, ea_var_orig = ea.copy(), ea_var.copy()

    submask = options['submask']
    if os.path.exists(submask):
        submask = load_nii(submask).get_fdata()
    else:
        submask = np.ones_like(ea)

    ea = nd.grey_closing(ea, size=(3, 3, 3))
    ea_var = nd.grey_closing(ea_var, size=(3, 3, 3))
    ea = ea > options["t_bin"]
    output_scan = ea.copy()
    # ea = ea*submask

    morphed = nd.binary_opening(output_scan, iterations=1)
    morphed = nd.binary_fill_holes(morphed, structure=np.ones((5, 5, 5))).astype(int)

    morphed = morphed * submask
    pred_labels, _ = nd.label(morphed, structure=np.ones((3, 3, 3)))

    label_list = np.unique(pred_labels)
    num_elements_by_lesion = nd.labeled_comprehension(
        morphed, pred_labels, label_list, np.sum, float, 0
    )

    output_scan = np.zeros_like(morphed)
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > options["l_min"]:
            # assign voxels to output
            current_voxels = np.stack(np.where(pred_labels == l), axis=1)
            output_scan[
                current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]
            ] = 1

    pred_labels, _ = nd.label(output_scan, structure=np.ones((3, 3, 3)))
    label_list = np.unique(pred_labels)
    num_elements_by_lesion = nd.labeled_comprehension(
        output_scan, pred_labels, label_list, np.sum, float, 0
    )

    # assign voxel-level cluster-wise ranks based on confidence
    prob, conf, conf_sort, uncert, coords = assign_rank_io(
        scan,
        output_scan,
        ea_orig,
        ea_var_orig,
        options["header"],
        options,
        pred_labels,
        label_list,
    )

    stats = {
        "probability": prob,
        "confidence": conf / 100,
        "var": uncert,
        "rank": conf_sort,
        "id": str(scan),
        "coords": coords,
    }

    stat_df = pd.DataFrame.from_records(stats)
    return stat_df