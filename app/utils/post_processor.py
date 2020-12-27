import os, subprocess
import numpy as np
from utils.metrics import *
from sklearn.metrics import cohen_kappa_score
from scipy import ndimage as nd
from scipy.ndimage import binary_opening, binary_dilation
from nibabel import load as load_nii
import nibabel as nib

def post_processing(input_scan, options, affine, header, save_nifti=True):
    """
    Post-process the probabilistic segmentation using parameters t_bin and l_min
    t_bin: threshold to binarize the output segmentations
    l_min: minimum lesion volume

    Inputs:
    - input_scan: probabilistic input image (segmentation)
    - options dictionary
    - save_nifti: save the result as nii

    Output:
    - output_scan: final binarized segmentation
    """

    t_bin = options['t_bin']
    # t_bin = 0
    l_min = options['l_min']
    output_scan = np.zeros_like(input_scan)
    labels_scan = np.zeros_like(input_scan)

    # threshold input segmentation
    t_segmentation = input_scan > t_bin
    # t_segmentation = input_scan > 0

    # perform morphological operations (dilation of the erosion of the input)
    morphed = binary_opening(t_segmentation, iterations=1)
    # morphed = t_segmentation
    # label connected components
    morphed = nd.binary_fill_holes(morphed, structure=np.ones((5,5,5))).astype(int)
    # pred_labels, num_labels = nd.label(morphed)
    pred_labels, num_labels = nd.label(morphed, structure=np.ones((3,3,3)))
    label_list = np.unique(pred_labels)
    num_elements_by_lesion = nd.labeled_comprehension(morphed, pred_labels, label_list, np.sum, float, 0)

    # filter candidates by size and store those > l_min
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l]>l_min:
            # assign voxels to output
            current_voxels = np.stack(np.where(pred_labels == l), axis=1)
            output_scan[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = 1

    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l]>l_min:
            # assign voxels to output
            current_voxels = np.stack(np.where(pred_labels == l), axis=1)
            labels_scan[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = num_elements_by_lesion[l].astype(np.int)


    count = np.count_nonzero(num_elements_by_lesion.astype(dtype=np.int) > l_min)

    options['test_morph_name'] = options['experiment'] + '_' + options['test_scan'] + '_out_morph_labels.nii.gz'

    # extract cluster information
    # from nilearn.regions import connected_label_regions
    # cluster_nii_tmp = nib.Nifti1Image(output_scan, np.eye(4))
    # cluster_nii = connected_label_regions(cluster_nii_tmp)
    # clusters = cluster_nii.get_data()
    # num_clusters = len(np.unique(labels)) - 1

    # options['test_clus_name'] = options['experiment'] + '_out_clus.nii.gz'

    #save the output segmentation as nifti
    if save_nifti:
        nii_out = nib.Nifti1Image(output_scan, affine, header)
        nii_out.to_filename(os.path.join(options['pred_folder'], options['test_name']))
        in_nii = os.path.join(options['pred_folder'], options['test_name'])
        out_mnc = os.path.join(options['pred_folder'], str.replace(options['test_name'], 'nii.gz', 'mnc'))
        subprocess.call(["/data/noel/noel2/local/brainvisa-Mandriva-2008.0-x86_64-4.1.0-2011_05_16/bin/AimsFileConvert", "-i", in_nii, "-o", out_mnc])
        # cluster_nii.to_filename(os.path.join(options['test_folder'], options['test_scan'], options['experiment'], options['test_clus_name']))
        labels_out = nib.Nifti1Image(labels_scan, affine, header)
        labels_out.to_filename(os.path.join(options['pred_folder'], options['test_morph_name']))
        in_nii = os.path.join(options['pred_folder'], options['test_morph_name'])
        out_mnc = os.path.join(options['pred_folder'], str.replace(options['test_morph_name'], 'nii.gz', 'mnc'))
        subprocess.call(["/data/noel/noel2/local/brainvisa-Mandriva-2008.0-x86_64-4.1.0-2011_05_16/bin/AimsFileConvert", "-i", in_nii, "-o", out_mnc])

    return output_scan, pred_labels, count


def extract_lesional_clus(label, input_scan, scan, options):
# find cluster component in the predicition corresponding to the true label cluster

    t_bin = options['t_bin']
    # t_bin = 0
    l_min = options['l_min']
    output_scan = np.zeros_like(input_scan)

    # threshold input segmentation
    t_segmentation = input_scan > t_bin
    # t_segmentation = input_scan > 0

    # perform morphological operations (dilation of the erosion of the input)
    morphed = binary_opening(t_segmentation, iterations=1)
    # morphed = t_segmentation
    # label connected components
    morphed = nd.binary_fill_holes(morphed, structure=np.ones((5,5,5))).astype(int)
    # pred_labels, num_labels = nd.label(morphed)

    pred_labels, num_labels = nd.label(morphed, structure=np.ones((3,3,3)))
    label_list = np.unique(pred_labels)
    num_elements_by_lesion = nd.labeled_comprehension(morphed, pred_labels, label_list, np.sum, float, 0)

    Y = np.zeros((len(num_elements_by_lesion > l_min)))
    for l in range(len(num_elements_by_lesion > l_min)):
        # Y[l] = np.linalg.norm(label - (pred_labels == l))
        Y[l] = dc(label, (pred_labels == l))

    # if Y.max() != 0:
    clus_ind = np.where(Y == Y.max())
    lesion_pred = np.copy(pred_labels)
    lesion_pred[lesion_pred != clus_ind] = 0
    lesion_pred[lesion_pred == clus_ind] = 1

    lesion_pred_out = nib.Nifti1Image(lesion_pred, np.eye(4))
    options['test_lesion_pred'] = options['experiment'] + '_' + options['test_scan'] + '_out_lesion_pred_only.nii.gz'
    lesion_pred_out.to_filename(os.path.join(options['pred_folder'], options['test_lesion_pred']))

    return lesion_pred


def performancer(perf, scan, test, label, lesion_pred, count):
    perf[scan] = perf_measure_vox(test.flatten(), label.flatten())
    perf[scan]['accuracy'] = accuracy_score(label.flatten(), test.flatten())
    perf[scan]['kappa'] = cohen_kappa_score(label.flatten(), lesion_pred.flatten())
    # perf[scan]['jaccard'] = jc(label.flatten(), lesion_pred.flatten())
    perf[scan]['dice_coef'] = dc(lesion_pred, label)

    cm = confusion_matrix(label.flatten(), test.flatten(), (1,0))
    cm_norm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_norm.astype(int))

    print('-'*80)
    print("dice coefficient: %.4f " %(perf[scan]['dice_coef']))
    print('-'*80)

    print('-'*80)
    perf[scan]['clusters'] = count
    print("no. of clusters (lesional + extra-lesional): %i " %(count))
    print('-'*80)

    return perf
