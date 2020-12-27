# from __future__ import print_function
import os, re, gc
import subprocess
import numpy as np
import pandas as pd

from nibabel import load as load_nii
import nibabel as nib

import _pickle as cPickle
import copy
import time
from keras.models import load_model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LambdaCallback
from keras.utils.np_utils import to_categorical
from keras.utils.io_utils import HDF5Matrix

# from pynvml import *

import json
from sklearn.model_selection import LeaveOneGroupOut

from utils.patch_dataloader import *
from utils.post_processor import *


def partition_leave_one_site_out(datafile='data_site_scanner_labels.xlsx', test_site=None):
    data = pd.read_excel(datafile)
    ids = data['index']
    groups = data['testing_dataset'].values

    logo = LeaveOneGroupOut()
    logo.get_n_splits(ids, ids, groups)

    folds = {}
    for train_index, test_index in logo.split(ids, ids, groups):
        test_group = data.train_site_label[test_index[0]]
        folds[test_group] = {}
        folds[test_group]['train_idx'], folds[test_group]['test_idx'] = train_index, test_index
        folds[test_group]['train_pids'], folds[test_group]['test_pids'] = ids[train_index].values, ids[test_index].values

    train, test = folds[test_site]['train_pids'], folds[test_site]['test_pids']

    return train, test, folds

def create_dataset(data_path, X, y):
    import h5py
    f = h5py.File(data_path, 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('data', X.shape, dtype='f')
    X_dset[:] = X
    # Creating dataset to store labels
    y_dset = f.create_dataset('labels', y.shape, dtype='i')
    y_dset[:] = y
    f.close()

def train_model(model, train_x_data, train_y_data, options):
    """
    Train the model using a cascade of two CNN

    inputs:

    - CNN model: a list containing the two cascaded CNN models

    - train_x_data: a nested dictionary containing training image paths:
           train_x_data['scan_name']['modality'] = path_to_image_modality

    - train_y_data: a dictionary containing labels
        train_y_data['scan_name'] = path_to_label

    - options: dictionary containing general hyper-parameters:


    Outputs:
        - trained model: list containing the two cascaded CNN models after training
    """
    if options['hostname'].startswith("hamlet"):
        # batch_size = options['mini_batch_size']
        batch_size = int(options['mini_batch_size']/2)
    else:
        batch_size = int(options['mini_batch_size']/2)
	# RAND = np.array_str(np.random.randint(1000,size=1))
    pwd = os.getcwd()
    RAND = time.strftime('%a''_' '%H_%M_%S')
    net_logs = os.path.join(options['weight_paths'], 'logs')
    if not os.path.exists(net_logs):
        # os.mkdir(options['weight_paths'])
        os.mkdir(os.path.join(options['weight_paths'], 'checkpoints'))
        os.mkdir(net_logs)

    # first iteration (CNN1):
    net_model = 'model_1'

    # tensorboard_cb = TensorBoard(log_dir='./logs/tensorboard', histogram_freq=0, write_graph=True, write_images=True, update_freq='batch')
    early_stopping_monitor = EarlyStopping(patience=options['patience'])
    net_weights = os.path.join(options['weight_paths'], 'checkpoints') + '/' + net_model + '_weights.h5'
    model_checkpoint = ModelCheckpoint(net_weights, monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger(net_logs + '/training_' + options['experiment'] + '_' + net_model + '_' + RAND + '_adadelta_log.csv')
    json_log = open('{:}/checkpoint_1.json'.format(os.path.join(options['weight_paths'], 'checkpoints')), mode='wt', buffering=1)
    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({'epoch': epoch, 'val_loss': logs['val_loss']}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )

    if os.path.isfile(net_weights) and options['load_checkpoint_1']:
        print("loading trained DNN1, model[0]: {} exists".format(net_weights))
        model[0] = load_model(net_weights)
    else:
        print('\n\n')
        print('====> DNN1 // loading training data')
        datapath = os.path.join(options['hdf5_data_dir'], options['experiment'] + '_LoSO_data.h5')
        if os.path.isfile(datapath):
            print(datapath + " exists, loading now")
            # datapath = options['data_path']
            n_patches = HDF5Matrix(datapath, 'labels').shape[0]
            start, end = [0, int(n_patches * (1-options['train_split']))], [int(n_patches * (1-options['train_split'])), n_patches]
            X, labels = HDF5Matrix(datapath, 'data', start=start[0], end=end[0]), HDF5Matrix(datapath, 'labels', start=start[0], end=end[0])
            X_val, y_val = HDF5Matrix(datapath, 'data',start=start[1], end=end[1]), HDF5Matrix(datapath, 'labels', start=start[1], end=end[1])
            print( '\n\n' )
            print( '====> DNN1 // fitting model', '\n' )
            print( '====> # 3D training patches:', X.shape[0] ,'\n' )
            print( '====> # patch size:', (X.shape[2],X.shape[3],X.shape[4]) ,'\n' )
            print( '====> # modalities:', (X.shape[1]) ,'\n' )

            model[0].fit(X, labels, batch_size=batch_size, epochs=options['max_epochs_1'], verbose=2, shuffle="batch", validation_data=(X_val,y_val), callbacks=[early_stopping_monitor, model_checkpoint, csv_logger, json_logging_callback])
        else:
            X, Y = load_training_data(train_x_data, train_y_data, options, subcort_masks=None)
            labels = to_categorical(Y, num_classes=2)
            print("\n hdf5 dataset is being created: {}".format(datapath))
            create_dataset(datapath, X, labels)
            print( '\n\n' )
            print( '====> DNN1 // fitting model', '\n' )
            print( '====> # 3D training patches:', X.shape[0] ,'\n' )
            print( '====> # patch size:', (X.shape[2],X.shape[3],X.shape[4]) ,'\n' )
            print( '====> # modalities:', (X.shape[1]) ,'\n' )

            model[0].fit(X, labels, batch_size=batch_size, epochs=options['max_epochs_1'], verbose=2, shuffle=True, validation_split=options['train_split'], callbacks=[early_stopping_monitor, model_checkpoint, csv_logger, json_logging_callback])

        copy_most_recent_model(os.path.join(options['weight_paths'], 'checkpoints'), net_model)
    # second iteration (CNN2):
    # load training data based on CNN1 candidates
    if options['train_2']:
        net_model = 'model_2'
        # tensorboard_cb = TensorBoard(log_dir='./logs/tensorboard', histogram_freq=0, write_graph=True, write_images=True, update_freq='batch')
        net_weights = os.path.join(options['weight_paths'], 'checkpoints') + '/' + net_model + '_weights.h5'
        model_checkpoint = ModelCheckpoint(net_weights, monitor='val_loss', save_best_only=True)
        early_stopping_monitor = EarlyStopping(patience=options['patience']+10)
        csv_logger = CSVLogger(net_logs + '/training_' + options['experiment'] + '_' + net_model + '_' + RAND + '_adadelta_log.csv')
        json_log = open('{:}/checkpoint_2.json'.format(os.path.join(options['weight_paths'], 'checkpoints')), mode='wt', buffering=1)
        json_logging_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: json_log.write(
                json.dumps({'epoch': epoch, 'val_loss': logs['val_loss']}) + '\n'),
            on_train_end=lambda logs: json_log.close()
        )

        datapath = os.path.join(options['hdf5_data_dir'], options['experiment'] + '_LoSO_data_intermediate.h5')
        if os.path.isfile(net_weights) and options['load_checkpoint_2']:
            print("loading DNN2, model[1]: {} exists".format(net_weights))
            model[1] = load_model(net_weights)
            if options['continue_training_2'] and os.path.isfile(datapath):
                print(datapath + " exists, loading now")
                # datapath = options['data_path']
                print( '====> DNN2 // loading training data from HDF5 dataset' )
                n_patches = HDF5Matrix(datapath, 'labels').shape[0]
                start, end = [0, int(n_patches * (1-options['train_split']))], [int(n_patches * (1-options['train_split'])), n_patches]
                X, labels = HDF5Matrix(datapath, 'data', start=start[0], end=end[0]), HDF5Matrix(datapath, 'labels', start=start[0], end=end[0])
                X_val, y_val = HDF5Matrix(datapath, 'data',start=start[1], end=end[1]), HDF5Matrix(datapath, 'labels', start=start[1], end=end[1])
                print( '\n\n' )
                print( '====> DNN2 // fitting model', '\n' )
                print( '====> # 3D training patches:', X.shape[0] ,'\n' )
                print( '====> # patch size:', (X.shape[2],X.shape[3],X.shape[4]) ,'\n' )
                print( '====> # modalities:', (X.shape[1]) ,'\n' )
                model[1].fit(X, labels, batch_size=batch_size, initial_epoch=options['initial_epoch_2'], epochs=options['max_epochs_2'], verbose=2, shuffle="batch", validation_data=(X_val,y_val), callbacks=[early_stopping_monitor, model_checkpoint, csv_logger, json_logging_callback])
            elif options['continue_training_2']:
                X, Y = load_training_data(train_x_data, train_y_data, options, model=model[0], subcort_masks=None)
                labels = to_categorical(Y, num_classes=2)
                model[1].fit(X, labels, batch_size=batch_size, initial_epoch=options['initial_epoch_2'], epochs=options['max_epochs']+50, verbose=2, shuffle=True, validation_split=options['train_split'], callbacks=[early_stopping_monitor, model_checkpoint, csv_logger, json_logging_callback])
            else:
                pass
        else:
            if os.path.isfile(datapath):
                print(datapath + " exists, loading now")
                # datapath = options['data_path']
                print( '====> DNN2 // loading training data from HDF5 dataset' )
                n_patches = HDF5Matrix(datapath, 'labels').shape[0]
                start, end = [0, int(n_patches * (1-options['train_split']))], [int(n_patches * (1-options['train_split'])), n_patches]
                X, labels = HDF5Matrix(datapath, 'data', start=start[0], end=end[0]), HDF5Matrix(datapath, 'labels', start=start[0], end=end[0])
                X_val, y_val = HDF5Matrix(datapath, 'data',start=start[1], end=end[1]), HDF5Matrix(datapath, 'labels', start=start[1], end=end[1])
                print( '\n\n' )
                print( '====> DNN2 // fitting model', '\n' )
                print( '====> # 3D training patches:', X.shape[0] ,'\n' )
                print( '====> # patch size:', (X.shape[2],X.shape[3],X.shape[4]) ,'\n' )
                print( '====> # modalities:', (X.shape[1]) ,'\n' )
                model[1].fit(X, labels, batch_size=batch_size, epochs=options['max_epochs_2'], verbose=2, shuffle="batch", validation_data=(X_val,y_val), callbacks=[early_stopping_monitor, model_checkpoint, csv_logger, json_logging_callback])
            else:
                X, Y = load_training_data(train_x_data, train_y_data, options, model=model[0], subcort_masks=None)
                labels = to_categorical(Y, num_classes=2)
                print("\n HDF5 dataset is being created: {}".format(datapath))
                create_dataset(datapath, X, labels)
                print( '\n\n' )
                print( '====> DNN2 // fitting model', '\n' )
                print( '====> # 3D training patches:', X.shape[0] ,'\n' )
                print( '====> # patch size:', (X.shape[2], X.shape[3], X.shape[4]) ,'\n' )
                print( '====> # modalities:', (X.shape[1]) ,'\n' )

                model[1].fit(X, labels, batch_size=batch_size, epochs=options['max_epochs_2'], verbose=2, shuffle=True, validation_split=options['train_split'], callbacks=[early_stopping_monitor, model_checkpoint, csv_logger, json_logging_callback])

            copy_most_recent_model(os.path.join(options['weight_paths'], 'checkpoints'), net_model)
    return model



def test_model(model, test_x_data, options, uncertainty=False):

    print("testing the model for scan: {}".format(test_x_data.keys()))

    threshold = options['th_dnn_train_2']
    # organize experiments
    # exp_folder = os.path.join(options['test_folder'], options['test_scan'], options['experiment'])
    # if not os.path.exists(exp_folder):
    #     os.mkdir(exp_folder)

    # first network
    options['test_name'] = options['experiment'] + '_prob_0.nii.gz'
    options['test_mean_name'] = options['experiment'] + '_prob_mean_0.nii.gz'
    options['test_var_name'] = options['experiment'] + '_prob_var_0.nii.gz'

    t1, _, _ = test_scan(model[0], test_x_data, options, save_nifti=True, uncertainty=True, T=20)
    # t1 = load_nii(os.path.join(options['pred_folder'], options['test_mean_name'])).get_data()

    # second network
    options['test_name'] = options['experiment'] + '_prob_1.nii.gz'
    options['test_mean_name'] = options['experiment'] + '_prob_mean_1.nii.gz'
    options['test_var_name'] = options['experiment'] + '_prob_var_1.nii.gz'
    t2, affine, header = test_scan(model[1], test_x_data, options, save_nifti=True, uncertainty=True, T=50, candidate_mask=t1>threshold)
    #
    # postprocess the output segmentation
    options['test_name'] = options['experiment'] + '_out_CNN.nii.gz'
    out_segmentation, lpred, count = post_processing(t2, options, affine, header, save_nifti=True)

    return t1, t2, out_segmentation, lpred, count


def test_scan(model, test_x_data, options, transit=None, save_nifti=False, uncertainty=False, candidate_mask=None, T=20):
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
    # nvmlInit()
    # handle = nvmlDeviceGetHandleByIndex(0)
    # info = nvmlDeviceGetMemoryInfo(handle)
    # bsize = info.total/1024/1024
    # # print( "total GPU memory available: %d MB" ) % (bsize)
    # if bsize < 2000:
    #     batch_size = 384
    #     print( "reducing batch_size to : {}".format(batch_size))
    #     options['batch_size'] = 100352
    # else:
    if options['hostname'].startswith("hamlet"):
        # batch_size = 2200
        batch_size = 2000
        options['batch_size'] = 350000
    if options['hostname'].startswith("silius") or options['hostname'].startswith("pandarus"):
        # batch_size = 2200
        batch_size = 2000
    else:
        # batch_size = 2800
        batch_size = 2000

    # get_scan name and create an empty nii image to store segmentation
    scans = test_x_data.keys()
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0]).get_data()
    header = load_nii(flair_scans[0]).header
    affine = header.get_qform()
    seg_image = np.zeros_like(flair_image)
    var_image = np.zeros_like(flair_image)

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
        if uncertainty:
            print("predicting uncertainty")
            # progbar = Progbar(target=len(batch))
            # pbar.update(1)
            y_pred, y_pred_var = predict_uncertainty(model, batch, batch_size=batch_size, T=T)
            # progbar.update(len(batch))
            # i+=1
        else:
            y_pred = model.predict(np.squeeze(batch), batch_size=batch_size, verbose=1)
            # pbar.update(1)
        # pbar.close()
        # y_pred = model.predict_on_batch(np.squeeze(batch))
        # model.evaluate(np.squeeze(batch), batch_size=256, verbose=1)
        [x, y, z] = np.stack(centers, axis=1)
        seg_image[x, y, z] = y_pred[:, 1]
        if uncertainty:
            var_image[x, y, z] = y_pred_var[:, 1]
    # progbar.update(len(test_x_data))

    # if not os.path.exists(options['pred_folder']):
    #     os.mkdir(options['pred_folder'])
    if save_nifti:
        # out_scan = nib.Nifti1Image(seg_image, np.eye(4))
        out_scan = nib.Nifti1Image(seg_image, affine, header)
        out_scan.to_filename(os.path.join(options['pred_folder'], options['test_mean_name']))
        in_nii = os.path.join(options['pred_folder'], options['test_mean_name'])
        out_mnc = os.path.join(options['pred_folder'], str.replace(options['test_mean_name'], 'nii.gz', 'mnc'))
        # subprocess.call(["/data/noel/noel2/local/brainvisa-Mandriva-2008.0-x86_64-4.1.0-2011_05_16/bin/AimsFileConvert", "-i", in_nii, "-o", out_mnc])

        if uncertainty:
            out_scan = nib.Nifti1Image(var_image, affine, header)
            out_scan.to_filename(os.path.join(options['pred_folder'], options['test_var_name']))
            in_nii = os.path.join(options['pred_folder'], options['test_var_name'])
            out_mnc = os.path.join(options['pred_folder'], str.replace(options['test_var_name'], 'nii.gz', 'mnc'))
            # subprocess.call(["/data/noel/noel2/local/brainvisa-Mandriva-2008.0-x86_64-4.1.0-2011_05_16/bin/AimsFileConvert", "-i", in_nii, "-o", out_mnc])


    if transit is not None:
        # test_folder = str.replace(test_folder, 'brain', 'predictions')
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        # out_scan = nib.Nifti1Image(seg_image, np.eye(4))
        out_scan = nib.Nifti1Image(seg_image, affine, header)
        test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_mean_0.nii.gz'
        out_scan.to_filename(os.path.join(test_folder, test_name))

        if uncertainty:
            out_scan = nib.Nifti1Image(var_image, affine, header)
            test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_var_0.nii.gz'
            out_scan.to_filename(os.path.join(test_folder, test_name))
        # in_nii = os.path.join(test_folder, options['experiment'], options['test_name'])
        # out_mnc = os.path.join(test_folder, options['experiment'], str.replace(options['test_name'], 'nii', 'mnc'))
        # subprocess.call(["/data/noel/noel2/local/brainvisa-Mandriva-2008.0-x86_64-4.1.0-2011_05_16/bin/AimsFileConvert", "-i", in_nii, "-o", out_mnc])

        # test_folder = str.replace(test_folder, 'brain', 'predictions')
        if not os.path.exists(os.path.join(test_folder, options['experiment'])):
            os.mkdir(os.path.join(test_folder, options['experiment']))

        out_scan = nib.Nifti1Image(seg_image, affine, header)
        #out_scan.to_filename(os.path.join(options['test_folder'], options['test_scan'], options['experiment'], options['test_name']))
        test_name = str.replace(scan, '_flair.nii.gz', '') + '_out_pred_0.nii.gz'
        out_scan.to_filename(os.path.join(test_folder, test_name))

    return seg_image, affine, header


def copy_most_recent_model(path, net_model):
    from shutil import copyfile
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if basename.endswith('.h5')]
    latest_model = max(paths, key=os.path.getctime)
    copyfile(latest_model, os.path.join(path, net_model)+'.h5')
