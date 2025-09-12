from operator import add, itemgetter

import h5py
import numpy as np
from nibabel import load as load_nii
from scipy.ndimage import binary_dilation
from tqdm import tqdm
from tqdm.contrib import tzip

from patch_dataloader import (binarize_label_gm,
                               select_voxels_from_previous_model)


def create_dataset(data_path, X, y):
    """
    Load train patches with size equal to patch_size, given a list of selected voxels

    Inputs:
       - X: training X data matrix for the particular channel [num_samples, p1, p2, p3]
       - y: training y labels [num_samples,]

    Outputs:
       - data_path: compressed HDF5 dataset with X and y
    """
    with h5py.File(data_path, "w") as f:
        # f = h5py.File(data_path, 'w')
        # Creating dataset to store features
        X_dset = f.create_dataset(
            "data",
            X.shape,
            dtype="f",
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        X_dset[:] = X
        # Creating dataset to store labels
        y_dset = f.create_dataset(
            "labels",
            y.shape,
            dtype="i8",
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        y_dset[:] = y
        # f.close()


def load_train_patches(
    x_data,
    y_data,
    selected_voxels,
    patch_size,
    subcort_masks=None,
    n_patches=1000,
    seed=666,
    datatype=np.float32,
):
    """
    Load train patches with size equal to patch_size, given a list of selected voxels

    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - y_data: list containing all subject image paths for the labels
       - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
       - patch_size: tuple containing patch size, 3D (p1, p2, p3)

    Outputs:
       - X: train X data matrix for the particular channel [num_samples, p1, p2, p3]
       - Y: train Y labels [num_samples, p1, p2, p3]
    """

    # load images and normalize their intensties
    images = [
        load_nii(name).get_fdata() for name in tqdm(x_data, desc="loading MRI images")
    ]
    images_norm = [
        (im.astype(dtype=datatype) - im[np.nonzero(im)].mean())
        / im[np.nonzero(im)].std()
        for im in tqdm(images, desc="normalize MRI intensities")
    ]
    del images

    # load labels
    lesion_masks = [
        binarize_label_gm(load_nii(name).get_fdata())
        for name in tqdm(y_data, desc="loading lesion labels")
    ]  # preserve only the GM component, ignore, WM and transmantle sign

    # load subcortical masks to exclude these voxels from training
    if subcort_masks is not None:
        submasks = [
            load_nii(name).get_fdata()
            for name in tqdm(subcort_masks, desc="load subcortical masks")
        ]
        nolesion_masks = [
            np.logical_and(np.logical_not(lesion), submask, brain)
            for lesion, submask, brain in tzip(
                lesion_masks,
                submasks,
                selected_voxels,
                desc="extract nonlesional masks",
            )
        ]
        del submasks
    else:
        nolesion_masks = [
            np.logical_and(np.logical_not(binary_dilation(lesion, iterations=5)), brain)
            for lesion, brain in tzip(
                lesion_masks, selected_voxels, desc="extract nonlesional masks"
            )
        ]

    # lesional_vox = 0
    # for lesion in lesion_masks:
    #     lesion_size = np.sum(lesion)
    #     lesional_vox += lesion_size
    #     if lesion_size < 1000:
    #         print("\nlesion_size: {}".format(lesion_size))
    # print("\ntotal lesional voxels: {}".format(lesional_vox))
    
    # Get all the x,y,z coordinates for each image
    lesion_centers = [
        get_mask_voxels(mask)
        for mask in tqdm(lesion_masks, desc="extract lesional coords")
    ]
    nolesion_centers = [
        get_mask_voxels(mask)
        for mask in tqdm(nolesion_masks, desc="extract nonlesional coords")
    ]
    del nolesion_masks

    # load all positive samples (lesional voxels) up to a maximum of n_patches
    np.random.seed(seed)
    indices = [
        np.random.permutation(range(0, len(center_les))).tolist()[
            : min(n_patches, len(center_les))
        ]
        for center_les in lesion_centers
    ]

    lesion_small = [
        itemgetter(*idx)(centers) for centers, idx in zip(nolesion_centers, indices)
    ]
    x_pos_patches = [
        np.array(get_patches(image, centers, patch_size))
        for image, centers in tzip(images_norm, lesion_small, desc="extract positive patches")
    ]
    # y_pos_patches = [
    #     np.array(get_patches(image, centers, patch_size))
    #     for image, centers in tzip(lesion_masks, lesion_small, desc="extract positive patch labels")
    # ]

    # load as many random negatives (non-lesions) samples as positive (lesions) samples
    indices = [
        np.random.permutation(range(0, len(center_no_les))).tolist()[
            : min(n_patches, len(center_les))
        ]
        for center_no_les, center_les in zip(nolesion_centers, lesion_centers)
    ]

    nolesion_small = [
        itemgetter(*idx)(centers) for centers, idx in zip(nolesion_centers, indices)
    ]
    x_neg_patches = [
        np.array(get_patches(image, centers, patch_size))
        for image, centers in tzip(images_norm, nolesion_small,  desc="extract negative patches")
    ]
    # y_neg_patches = [
    #     np.array(get_patches(image, centers, patch_size))
    #     for image, centers in tzip(lesion_masks, nolesion_small, desc="extract negative patch labels")
    # ]

    # concatenate positive and negative patches for each subject
    X = np.concatenate(
        [np.concatenate([x1, x2]) for x1, x2 in zip(x_pos_patches, x_neg_patches)], axis=0
        )
    # Y = np.concatenate(
    #     [np.concatenate([y1, y2]) for y1, y2 in zip(y_pos_patches, y_neg_patches)], axis=0
    #     )
    Y = np.concatenate(
        [np.concatenate([np.ones(y1.shape[0]), np.zeros(y2.shape[0])]) for y1, y2 in zip(x_pos_patches, x_neg_patches)], axis=0
        )

    return X, Y


def load_training_data(train_x_data, train_y_data, options, subcort_masks, model=None):
    """
    Load training and label samples for all given scans and modalities.

    Inputs:

    train_x_data: a nested dictionary containing training image paths:
        train_x_data['scan_name']['modality'] = path_to_image_modalities

    train_y_data: a dictionary containing labels
        train_y_data['scan_name'] = path_to_labels

    options: dictionary containing general hyper-parameters:
        - options['min_th'] = min threshold to remove voxels for training
        - options['patch_size'] = tuple containing patch size, 3D (p1, p2, p3)
        - options['randomize_train'] = randomize/shuffle the data

    model: CNN model used to select training candidates

    Outputs:
        - X: np.array [num_samples, num_channels, p1, p2, p2]
        - Y: np.array [num_samples, 1]

    """

    # get_scan names and number of modalities used
    scans = list(train_x_data.keys())
    modalities = train_x_data[scans[0]].keys()

    # select voxels for training:
    #   if no model is passed, training samples are extracted by discarding the CSF and darker WM in FLAIR, and use all remaining voxels.
    #   if model is passed, use the trained model to extract all voxels with probability > 0.1
    if model is None:
        flair_scans = [train_x_data[s]["FLAIR"] for s in scans]
        selected_voxels = select_training_voxels(flair_scans, options["min_th"])
    else:
        selected_voxels = select_voxels_from_previous_model(
            model, train_x_data, options
        )

    # extract patches and labels for each of the modalities
    data = []

    for m in modalities:
        x_data = [train_x_data[s][m] for s in scans]
        y_data = [train_y_data[s] for s in scans]
        if subcort_masks is not None:
            submasks = [subcort_masks[s] for s in scans]
            x_patches, y_patches = load_train_patches(
                x_data,
                y_data,
                selected_voxels,
                options["patch_size"],
                submasks,
                n_patches=options["n_patches"],
            )
        else:
            x_patches, y_patches = load_train_patches(
                x_data,
                y_data,
                selected_voxels,
                options["patch_size"],
                subcort_masks=None,
                n_patches=options["n_patches"],
            )
        print("{} shape: {}".format(m, x_patches.shape))
        data.append(x_patches)
    # stack patches along the channels' dimension [samples, channels, p1, p2, p3]
    X = np.stack(data, axis=1)
    y = y_patches

    # apply randomization if selected
    if options["randomize_train"]:
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        X = np.random.permutation(X.astype(dtype=np.float32))
        np.random.seed(seed)
        Y = np.random.permutation(y.astype(dtype=np.int8))
    else:
        X = X.astype(dtype=np.float32)
        Y = y.astype(dtype=np.int8)

    return X, Y


def get_mask_voxels(mask):
    """
    Compute x,y,z coordinates of a binary mask

    Input:
       - mask: binary mask

    Output:
       - list of tuples containing the (x,y,z) coordinate for each of the input voxels
    """

    indices = np.stack(np.nonzero(mask), axis=1)
    indices = [tuple(idx) for idx in indices]
    return indices


def get_patches(image, centers, patch_size=(16, 16, 16)):
    """
    Get image patches of arbitrary size based on a set of centers
    """
    # If the size has even numbers, the patch will be centered. If not, it will try to create an square almost centered.
    # By doing this we allow pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]

    if list_of_tuples and sizes_match:
        patch_half = tuple([idx // 2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        # padding = tuple((np.int(idx), np.int(size)-np.int(idx)) for idx, size in zip(patch_half, patch_size))
        padding = tuple((idx, size - idx) for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode="constant", constant_values=0)
        slices = [
            [
                slice(c_idx - p_idx, c_idx + (s_idx - p_idx))
                for (c_idx, p_idx, s_idx) in zip(center, patch_half, patch_size)
            ]
            for center in new_centers
        ]
        # patches = [new_image[idx] for idx in slices]
        patches = [new_image[tuple(idx)] for idx in slices]

    return patches


def select_training_voxels(input_masks, threshold=0.1, datatype=np.float32, t1=0):
    """
    Select voxels for training based on a intensity threshold

    Inputs:
        - input_masks: list containing all subject image paths for a single modality
        - threshold: minimum threshold to apply (after normalizing images with 0 mean and 1 std)

    Output:
        - rois: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
    """

    # load images and normalize their intensities
    images = [load_nii(image_name).get_fdata() for image_name in input_masks]
    images_norm = [
        (im.astype(dtype=datatype) - im[np.nonzero(im)].mean())
        / im[np.nonzero(im)].std()
        for im in images
    ]

    # select voxels with intensity higher than threshold
    rois = [
        image > threshold for image in tqdm(images_norm, desc="extract sampling masks from FLAIR thresholding")
    ]
    return rois


def binarize_label_gm(mask):
    # discard labels wm (2) and transmantle sign (6)
    mask_ = np.zeros_like(mask)
    tmp = np.stack(np.where(mask == 1), axis=1)
    mask_[tmp[:, 0], tmp[:, 1], tmp[:, 2]] = 1
    return mask_.astype(np.bool)
