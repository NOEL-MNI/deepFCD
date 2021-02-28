# from __future__ import print_function
import numpy as np
from scipy import ndimage as nd
from scipy.ndimage import binary_dilation
from nibabel import load as load_nii
import nibabel as nib
from operator import itemgetter, add
import os
from tqdm import trange
# from pynvml import *
from keras import backend as K
from utils.keras_bayes_utils import *


def test_scan_uncertainty(model, test_x_data, scan, options, intermediate=None, save_nifti=False, uncertainty=True, candidate_mask=None, T=20):
    """
    Test data based on one model
    Input:
    - test_x_data: a nested dictionary containing training image paths:
            train_x_data['scan_name']['modality'] = path_to_image_modality
    - save_nifti: save image segmentation
    - candidate_mask: a binary masks containing voxels to classify

    Output:
    - test_scan = Output image containing the probability output segmentation
    - If save_nifti --> Saves a nii file at specified location options['test_folder']/['test_scan']
    """
    if options['hostname'].startswith("hamlet"):
        # batch_size = 2200
        batch_size = 5120
        options['batch_size'] = 350000
    else:
        # batch_size = 2800
        batch_size = 2000

    # get_scan name and create an empty nii image to store segmentation
    tmp = {}
    tmp[scan] = test_x_data
    test_x_data = tmp
    scans = test_x_data.keys()
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0]).get_data()
    header = load_nii(flair_scans[0]).header
    affine = header.get_qform()

    # get test paths
    _, scan = os.path.split(flair_scans[0])
    test_folder = os.path.join('/host/silius/local_raid/ravnoor/01_Projects/55_Bayesian_DeepLesion_LoSo/data/predictions', options['experiment'])
    # test_folder = '/host/silius/local_raid/ravnoor/01_Projects/06_DeepLesion_LoSo/data/predictions
    if not os.path.exists(test_folder):
        # os.path.join(test_folder, options['experiment'])
        os.mkdir(test_folder)

    print('-'*60)
    print(str.replace(scan, '_flair.nii.gz', ''))
    print('-'*60)
    # compute lesion segmentation in batches of size options['batch_size']

    pred_fname = os.path.join(test_folder, str.replace(scan, '_flair.nii.gz', '') + '_out_pred_mean_0.nii.gz')
    if os.path.isfile(pred_fname):
        print("reading {} from disk".format(pred_fname))
        thresh_image = load_nii(pred_fname).get_data()
        # print(thresh_image.shape, np.max(thresh_image))
    else:
        seg_image = np.zeros_like(flair_image)
        var_image = np.zeros_like(flair_image)
        thresh_image = np.zeros_like(flair_image)
        for batch, centers in load_test_patches(test_x_data, options, options['patch_size'], options['batch_size'], options['min_th'], candidate_mask):
            print("predicting uncertainty")
            y_pred, y_pred_var = predict_uncertainty(model, batch, batch_size=batch_size, T=T)
            [x, y, z] = np.stack(centers, axis=1)
            seg_image[x, y, z] = y_pred[:, 1]
            var_image[x, y, z] = y_pred_var[:, 1]

        if intermediate is not None:
            # test_folder = str.replace(test_folder, 'brain', 'predictions')
            if not os.path.exists(test_folder):
                os.mkdir(test_folder)
            # out_scan = nib.Nifti1Image(seg_image, np.eye(4))
            out_scan = nib.Nifti1Image(seg_image, affine, header)
            test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_mean_0.nii.gz'
            out_scan.to_filename(os.path.join(test_folder, test_name))

            out_scan = nib.Nifti1Image(var_image, affine, header)
            test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_var_0.nii.gz'
            out_scan.to_filename(os.path.join(test_folder, test_name))

            # test_folder = str.replace(test_folder, 'brain', 'predictions')
            if not os.path.exists(os.path.join(test_folder, options['experiment'])):
                os.mkdir(os.path.join(test_folder, options['experiment']))

            out_scan = nib.Nifti1Image(seg_image, affine, header)
            #out_scan.to_filename(os.path.join(options['test_folder'], options['test_scan'], options['experiment'], options['test_name']))
            test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_0.nii.gz'
            out_scan.to_filename(os.path.join(test_folder, test_name))

            thresh_image = seg_image.copy()

    return thresh_image


def select_voxels_from_previous_model(model, train_x_data, options):
    """
    Select training voxels from image segmentation masks
    """
    threshold = options['th_dnn_train_2']
    # get_scan names
    scans = list(train_x_data.keys())
    # mask  = [test_scan_uncertainty(model, dict(train_x_data.items()[s:s+1]), options, intermediate=1, uncertainty=True)[0] > threshold for s in trange(len(scans), desc='sel_vox_prev_model_pred_mean')]
    mask  = [test_scan_uncertainty(model, dict(train_x_data[scans[s]]), scans[s], options, intermediate=1, uncertainty=True) > threshold for s in trange(len(scans), desc='sel_vox_prev_model_pred_mean')]

    return mask


def predict_uncertainty(model, data, batch_size, T=10):
    input = model.layers[0].input
    output = model.layers[-1].output
    f_stochastic = K.function([input, K.learning_phase()], output) # instantiates a Keras function.
    K.set_image_dim_ordering('th')
    K.set_image_data_format('channels_first')

    Yt_hat = np.array([predict_stochastic(f_stochastic, data, batch_size=batch_size) for _ in trange(T, ascii=True, desc="predict_stochastic")])
    MC_pred = np.mean(Yt_hat, 0)
    MC_pred_var = np.var(Yt_hat, 0)

    return MC_pred, MC_pred_var


def predict_stochastic(f, ins, batch_size=128, verbose=False):
    '''
        function to loop over some data in batches.
    '''
    nb_sample = len(ins)
    outs = []
    if verbose:
        progbar = Progbar(target=nb_sample)
    batches = make_batches(nb_sample, batch_size)
    index_array = np.arange(nb_sample)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        ins_batch = slice_X(ins, batch_ids)
        batch_outs = f([ins_batch, 1])
        if type(batch_outs) != list:
            batch_outs = [batch_outs]
        if batch_index == 0:
            for batch_out in batch_outs:
                shape = (nb_sample,) + batch_out.shape[1:]
                outs.append(np.zeros(shape))

        for i, batch_out in enumerate(batch_outs):
            outs[i][batch_start:batch_end] = batch_out
        if verbose:
            progbar.update(batch_end)
    return outs[0]


def load_training_data(train_x_data, train_y_data, options, subcort_masks, model=None):
    '''
    Load training and label samples for all given scans and modalities.

    Inputs:

    train_x_data: a nested dictionary containing training image paths:
        train_x_data['scan_name']['modality'] = path_to_image_modalities

    train_y_data: a dictionary containing labels
        train_y_data['scan_name'] = path_to_labels

    options: dictionary containing general hyper-parameters:
        - options['min_th'] = min threshold to remove voxels for training
        - options['size'] = tuple containing patch size, 3D (p1, p2, p3)
        - options['randomize_train'] = randomizes data

    model: CNN model used to select training candidates

    Outputs:
        - X: np.array [num_samples, num_channels, p1, p2, p2]
        - Y: np.array [num_samples, 1]

    '''

    # get_scan names and number of modalities used
    scans = list(train_x_data.keys())
    modalities = train_x_data[scans[0]].keys()

    # select voxels for training:
    #   if no model is passed, training samples are extracted by discarding the CSF and darker WM in FLAIR, and use all remaining voxels.
    #   if model is passed, use the trained model to extract all voxels with probability > 0.4
    if model is None:
        flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = select_training_voxels(flair_scans, options['min_th'])
    else:
        selected_voxels = select_voxels_from_previous_model(model, train_x_data, options)

    # extract patches and labels for each of the modalities
    data = []

    for m in modalities:
        x_data = [train_x_data[s][m] for s in scans]
        y_data = [train_y_data[s] for s in scans]
        if subcort_masks is not None:
            submasks = [subcort_masks[s] for s in scans]
            x_patches, y_patches = load_train_patches(x_data, y_data, selected_voxels, options['patch_size'], submasks)
        else:
            x_patches, y_patches = load_train_patches(x_data, y_data, selected_voxels, options['patch_size'], subcort_masks=None)
        data.append(x_patches)
    # stack patches in channels [samples, channels, p1, p2, p3]
    X = np.stack(data, axis = 1)
    Y = y_patches

    # apply randomization if selected
    if options['randomize_train']:
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        X = np.random.permutation(X.astype(dtype=np.float32))
        np.random.seed(seed)
        Y = np.random.permutation(Y.astype(dtype=np.int32))

    # Y = [num_samples,]
    if Y.shape[3] == 1:
        Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, :]
    else:
        Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, Y.shape[3] // 2]

    Y = np.squeeze(Y)

    return X, Y


def select_training_voxels(input_masks, threshold=0.5, datatype=np.float32, t1=0):
    """
    Select voxels for training based on a intensity threshold

    Inputs:
        - input_masks: list containing all subject image paths for a single modality
        - threshold: minimum threshold to apply (after normalizing images with 0 mean and 1 std)

    Output:
        - rois: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
    """

    # load images and normalize their intensities
    images = [load_nii(image_name).get_data() for image_name in input_masks]
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]

    # select voxels with intensity higher than threshold
    rois = [image > threshold for image in images_norm]
    return rois


def select_testing_voxels(flair, t1=None, threshold=0.5, datatype=np.float32):
    """
    Select voxels for training based on a intensity threshold

    Inputs:
        - input_masks: list containing all subject image paths for a single modality
        - threshold: minimum threshold to apply (after normalizing images with 0 mean and 1 std)

    Output:
        - rois: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
    """
    images = [load_nii(image_name).get_data() for image_name in flair]
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    rois = [image > threshold for image in images_norm]
    return rois


def load_train_patches(x_data, y_data, selected_voxels, patch_size, subcort_masks=None, seed=666, datatype=np.float32):
    """
    Load train patches with size equal to patch_size, given a list of selected voxels

    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - y_data: list containing all subject image paths for the labels
       - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
       - patch_size: tuple containing patch size, 3D (p1, p2, p3)

    Outputs:
       - X: Train X data matrix for the particular channel [num_samples, p1, p2, p3]
       - Y: Train Y labels [num_samples, p1, p2, p3]
    """

    # load images and normalize their intensties
    images = [load_nii(name).get_data() for name in x_data]
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]

    # load labels

    lesion_masks = [binarize_label_gm(load_nii(name).get_data()) for name in y_data] # preserve only the GM component, ignore, WM and transmantle sign

    # load subcortical masks to exclude these voxels from training
    if subcort_masks is not None:
        submasks = [load_nii(name).get_data() for name in subcort_masks]
        nolesion_masks = [np.logical_and(np.logical_not(lesion), submask, brain) for lesion, submask, brain in zip(lesion_masks, submasks, selected_voxels)]
    else:
        nolesion_masks = [np.logical_and(np.logical_not(binary_dilation(lesion, iterations=5)), brain) for lesion, brain in zip(lesion_masks, selected_voxels)]

    # Get all the x,y,z coordinates for each image
    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]

    # load all positive samples (lesional voxels) and the same number of random negatives samples
    np.random.seed(seed)

    x_pos_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(images_norm, lesion_centers)]
    y_pos_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(lesion_masks, lesion_centers)]

    indices = [np.random.permutation(range(0, len(center_no_les))).tolist()[:len(center_les)] for center_no_les, center_les in zip(nolesion_centers, lesion_centers)]

    nolesion_small = [itemgetter(*idx)(centers) for centers, idx in zip(nolesion_centers, indices)]
    x_neg_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(images_norm, nolesion_small)]
    y_neg_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(lesion_masks, nolesion_small)]

    # concatenate positive and negative patches for each subject
    X = np.concatenate([np.concatenate([x1, x2]) for x1, x2 in zip(x_pos_patches, x_neg_patches)], axis = 0)
    Y = np.concatenate([np.concatenate([y1, y2]) for y1, y2 in zip(y_pos_patches, y_neg_patches)], axis= 0)

    return X, Y


def binarize_label_gm(mask):
    # discard labels wm (2) and transmantle sign (6)
    mask_ = np.zeros_like(mask)
    tmp = np.stack(np.where(mask == 1), axis=1)
    mask_[tmp[:,0], tmp[:,1], tmp[:,2]] = 1
    # print("mask shape: {}, type: {}, unique: {}".format(mask_.shape, type(mask_), np.unique(mask_)))
    return mask_.astype(np.bool)


def load_test_patches(test_x_data, options, patch_size, batch_size, threshold, voxel_candidates=None, datatype=np.float32):
    """
    Function generator to load test patches with size equal to patch_size, given a list of selected voxels. Patches are
    returned in batches to reduce the amount of RAM used

    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
       - Voxel candidates: a binary mask containing voxels to select for testing

    Outputs (in batches):
       - X: Train X data matrix for the particular channel [num_samples, p1, p2, p3]
       - voxel_coord: list of tuples corresponding voxel coordinates (x,y,z) of selected patches
    """

    # get scan names and number of modalities used
    scans = list(test_x_data.keys())
    modalities = test_x_data[scans[0]].keys()

    # load all image modalities and normalize intensities
    images = []

    for m in modalities:
        raw_images = [load_nii(test_x_data[s][m]).get_data() for s in scans]
        images.append([(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in raw_images])

    # select voxels for testing. Discard CSF and darker WM in FLAIR.
    # If voxel_candidates is not selected, using intensity > 0.4 in FLAIR, else use
    # the binary mask to extract candidate voxels
    if voxel_candidates is None:
        flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = [get_mask_voxels(mask) for mask in select_training_voxels(flair_scans, threshold)][0]
    else:
        selected_voxels = get_mask_voxels(voxel_candidates)

    # yield data for testing with size equal to batch_size
    for i in range(0, len(selected_voxels), batch_size):
        c_centers = selected_voxels[i:i+batch_size]
        X = []
        for image_modality in images:
            X.append(get_patches(image_modality[0], c_centers, patch_size))
        yield np.stack(X, axis = 1), c_centers


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
        patch_half = tuple([idx//2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        # padding = tuple((np.int(idx), np.int(size)-np.int(idx)) for idx, size in zip(patch_half, patch_size))
        padding = tuple((idx, size-idx) for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = [[slice(c_idx-p_idx, c_idx+(s_idx-p_idx)) for (c_idx, p_idx, s_idx) in zip(center, patch_half, patch_size)] for center in new_centers]
        # patches = [new_image[idx] for idx in slices]
        patches = [new_image[tuple(idx)] for idx in slices]

    return patches
