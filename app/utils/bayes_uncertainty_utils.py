import os
import numpy as np
from tqdm import trange
from pynvml import *
from nibabel import load as load_nii
import nibabel as nib

from utils.patch_dataloader import *

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
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    bsize = info.total/1024/1024
    # print "total GPU memory available: %d MB" % (bsize)
    if bsize < 2000:
        batch_size = 384
        print("reducing batch_size to : {}".format(batch_size))
        options['batch_size'] = 100352
    else:
        if options['hostname'].startswith("hamlet"):
            # batch_size = 2200
            batch_size = 3000
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
    # affine = header.get_qform()
    seg_image = np.zeros_like(flair_image)
    var_image = np.zeros_like(flair_image)
    thresh_image = np.zeros_like(flair_image)

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
        out_scan = nib.Nifti1Image(seg_image, header=header)
        test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_mean_0.nii.gz'
        out_scan.to_filename(os.path.join(test_folder, test_name))

        out_scan = nib.Nifti1Image(var_image, header=header)
        test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_var_0.nii.gz'
        out_scan.to_filename(os.path.join(test_folder, test_name))

        # test_folder = str.replace(test_folder, 'brain', 'predictions')
        if not os.path.exists(os.path.join(test_folder, options['experiment'])):
            os.mkdir(os.path.join(test_folder, options['experiment']))

        out_scan = nib.Nifti1Image(seg_image, header=header)
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
    # print(scans)
    print(dict(train_x_data[scans[0]]))

    # mask  = [test_scan_uncertainty(model, dict(train_x_data.items()[s:s+1]), options, intermediate=1, uncertainty=True)[0] > threshold for s in trange(len(scans), desc='sel_vox_prev_model_pred_mean')]
    mask  = [test_scan_uncertainty(model, dict(train_x_data[scans[s]]), scans[s], options, intermediate=1, uncertainty=True)[0] > threshold for s in trange(len(scans), desc='sel_vox_prev_model_pred_mean')]

    return mask

def predict_uncertainty(model, data, batch_size, T=10):
    input = model.layers[0].input
    output = model.layers[-1].output
    f_stochastic = K.function([input, K.learning_phase()], output) # instantiates a Keras function.
    K.set_image_dim_ordering('th')
    K.set_image_data_format('channels_first')

    Yt_hat = np.array([predict_stochastic(f_stochastic, data, batch_size=batch_size) for _ in tqdm(xrange(T), ascii=True, desc="predict_stochastic")])
    MC_pred = np.mean(Yt_hat, 0)
    MC_pred_var = np.var(Yt_hat, 0)

    return MC_pred, MC_pred_var


def predict_stochastic(f, ins, batch_size=128, verbose=0):
    '''
        Abstract method to loop over some data in batches.
    '''
    nb_sample = len(ins)
    outs = []
    if verbose == 1:
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
        if verbose == 1:
            progbar.update(batch_end)
    return outs[0]
