# %%
import json
import os
import time
from shutil import copyfile

import ants
import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from bids import BIDSLayout

# %%
from keras.callbacks import CSVLogger, EarlyStopping, LambdaCallback, ModelCheckpoint
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from nibabel import load as load_nii
from sklearn.model_selection import LeaveOneGroupOut

from utils.patch_dataloader import *
from utils.post_processor import *
import re


# %%
def print_data_shape(X):
    """ Print shape of the training data

    Args:
        X (_type_): numpy array with the 3D patches for T1w and FLAIR
    """    
    print("====> # 3D training patches:", X.shape[0], "\n")
    print("====> # patch size:", (X.shape[2], X.shape[3], X.shape[4]), "\n")
    print("====> # modalities:", (X.shape[1]), "\n")


def partition_leave_one_site_out(datafile=None, test_site=None):
    """Partition the data based on leave-one-site-out (LoSo).

    Args:
        datafile (_type_, optional): CSV file containing all patient IDs associated with their respective site label. Defaults to None.
        test_site (_type_, optional): The site label to be used as held-out. Defaults to None.

    Returns:
        _type_: _description_
    """    
    data = pd.read_excel(datafile)
    ids = data["index"]
    groups = data["testing_dataset"].values
    logo = LeaveOneGroupOut()
    logo.get_n_splits(ids, ids, groups)

    folds = {}
    for train_index, test_index in logo.split(ids, ids, groups):
        test_group = data.train_site_label[test_index[0]]
        folds[test_group] = {}
        folds[test_group]["train_idx"], folds[test_group]["test_idx"] = (
            train_index,
            test_index,
        )
        folds[test_group]["train_pids"], folds[test_group]["test_pids"] = (
            ids[train_index].values,
            ids[test_index].values,
        )

    train, test = folds[test_site]["train_pids"], folds[test_site]["test_pids"]
    return train, test, folds


def create_dataset(datapath, X, y):
    f = h5py.File(datapath, "w")
    # create dataset to store features
    X_dset = f.create_dataset("data", X.shape, dtype="f")
    X_dset[:] = X
    # create dataset to store labels
    y_dset = f.create_dataset("labels", y.shape, dtype="i")
    y_dset[:] = y
    f.close()


def load_dataset(datapath, options):
    train_val_split = options["train_split"]
    n_patches = HDF5Matrix(datapath, "labels").shape[0]
    # get the train and validation patch indices
    start, end = [0, int(n_patches * (1 - train_val_split))], [
        int(n_patches * (1 - train_val_split)),
        n_patches,
    ]
    # extract the training dataset w/ labels
    X, y = HDF5Matrix(datapath, "data", start=start[0], end=end[0]), HDF5Matrix(
        datapath, "labels", start=start[0], end=end[0]
    )
    # extract the validation dataset w/ labels
    X_val, y_val = HDF5Matrix(datapath, "data", start=start[1], end=end[1]), HDF5Matrix(
        datapath, "labels", start=start[1], end=end[1]
    )
    return X, y, X_val, y_val


def model_callbacks(net_weights, net_model, net_logs, options, RAND):
    patience = options["patience"]
    f_checkpoint = "checkpoint_1"
    if net_model == "model_2":
        patience = patience + 10
        f_checkpoint = "checkpoint_2"
    early_stopping_monitor = EarlyStopping(patience=patience)
    model_checkpoint = ModelCheckpoint(
        net_weights, monitor="val_loss", save_best_only=True
    )
    csv_logger = CSVLogger(
        net_logs
        + "/training_"
        + options["experiment"]
        + "_"
        + net_model
        + "_"
        + RAND
        + "_adadelta_log.csv"
    )

    json_log = open(
        "{:}/{:}.json".format(
            os.path.join(options["weight_paths"], "checkpoints"), f_checkpoint
        ),
        mode="wt",
        buffering=1,
    )
    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({"epoch": epoch, "val_loss": logs["val_loss"]}) + "\n"
        ),
        on_train_end=lambda logs: json_log.close(),
    )
    return early_stopping_monitor, model_checkpoint, csv_logger, json_logging_callback


def train_model(model, train_x_data, train_y_data, options):
    """
    train the model using a cascade of two CNNs

    inputs:
    - CNN model: a list containing the two cascaded CNN models
    - train_x_data: a nested dictionary containing training image paths:
           train_x_data['scan_name']['modality'] = path_to_image_modality
    - train_y_data: a dictionary containing labels
        train_y_data['scan_name'] = path_to_label
    - options: dictionary containing general hyper-parameters:

    outputs:
        - trained model: list containing the two cascaded CNN models after training
    """
    batch_size = int(options["mini_batch_size"] / 2)
    RAND = time.strftime("%a" "_" "%H_%M_%S")
    net_logs = os.path.join(options["weight_paths"], "logs")
    if not os.path.exists(net_logs):
        os.mkdir(os.path.join(options["weight_paths"], "checkpoints"))
        os.mkdir(net_logs)

    # first iteration (CNN1):
    net_model = "model_1"
    net_weights = (
        os.path.join(options["weight_paths"], "checkpoints")
        + "/"
        + net_model
        + "_weights.h5"
    )

    # model training diagnostics + logging callbacks
    # tensorboard_cb = TensorBoard(log_dir='./logs/tensorboard', histogram_freq=0, write_graph=True, write_images=True, update_freq='batch')
    (
        early_stopping_monitor,
        model_checkpoint,
        csv_logger,
        json_logging_callback,
    ) = model_callbacks(net_weights, net_model, net_logs, options, RAND)

    if os.path.isfile(net_weights) and options["load_checkpoint_1"]:
        print("loading trained DNN1, model[0]: {} exists".format(net_weights))
        model[0] = load_model(net_weights)
    else:
        print("\n\n")
        print("====> DNN1 // loading training data")
        datapath = os.path.join(
            options["hdf5_data_dir"], options["experiment"] + "_LoSO_data.h5"
        )
        if os.path.isfile(datapath):
            print(datapath + " exists, loading now")
            # datapath = options['data_path']
            X, y, X_val, y_val = load_dataset(datapath, options)
            print("\n\n")

            print("====> DNN1 // fitting model", "\n")
            print_data_shape(X)

            model[0].fit(
                X,
                y,
                batch_size=batch_size,
                epochs=options["max_epochs_1"],
                verbose=2,
                shuffle="batch",
                validation_data=(X_val, y_val),
                callbacks=[
                    early_stopping_monitor,
                    model_checkpoint,
                    csv_logger,
                    json_logging_callback,
                ],
            )
        else:
            X, labels = load_training_data(
                train_x_data, train_y_data, options, subcort_masks=None
            )
            y = to_categorical(labels, num_classes=2)
            if options["save_as_hdf5"]:
                print("\n hdf5 dataset is being created: {}".format(datapath))
                create_dataset(datapath, X, y)
                print("\n\n")

            print("====> DNN1 // fitting model", "\n")
            print_data_shape(X)

            model[0].fit(
                X,
                y,
                batch_size=batch_size,
                epochs=options["max_epochs_1"],
                verbose=2,
                shuffle=True,
                validation_split=options["train_split"],
                callbacks=[
                    early_stopping_monitor,
                    model_checkpoint,
                    csv_logger,
                    json_logging_callback,
                ],
            )

        copy_most_recent_model(
            os.path.join(options["weight_paths"], "checkpoints"), net_model
        )
    # second iteration (CNN2):
    # load training data based on CNN1 candidates
    if options["train_2"]:
        net_model = "model_2"
        net_weights = (
            os.path.join(options["weight_paths"], "checkpoints")
            + "/"
            + net_model
            + "_weights.h5"
        )

        # model training diagnostics + logging callbacks
        # tensorboard_cb = TensorBoard(log_dir='./logs/tensorboard', histogram_freq=0, write_graph=True, write_images=True, update_freq='batch')
        (
            early_stopping_monitor,
            model_checkpoint,
            csv_logger,
            json_logging_callback,
        ) = model_callbacks(net_weights, net_model, net_logs, options, RAND)

        datapath = os.path.join(
            options["hdf5_data_dir"],
            options["experiment"] + "_LoSO_data_intermediate.h5",
        )
        if os.path.isfile(net_weights) and options["load_checkpoint_2"]:
            print("loading DNN2, model[1]: {} exists".format(net_weights))
            model[1] = load_model(net_weights)
            if options["continue_training_2"] and os.path.isfile(datapath):
                print(datapath + " exists, loading now")
                # datapath = options['data_path']
                print("====> DNN2 // loading training data from HDF5 dataset")
                X, y, X_val, y_val = load_dataset(datapath, options)
                print("\n\n")

                print("====> DNN2 // fitting model", "\n")
                print_data_shape(X)

                model[1].fit(
                    X,
                    y,
                    batch_size=batch_size,
                    initial_epoch=options["initial_epoch_2"],
                    epochs=options["max_epochs_2"],
                    verbose=2,
                    shuffle="batch",
                    validation_data=(X_val, y_val),
                    callbacks=[
                        early_stopping_monitor,
                        model_checkpoint,
                        csv_logger,
                        json_logging_callback,
                    ],
                )
            elif options["continue_training_2"]:
                X, labels = load_training_data(
                    train_x_data,
                    train_y_data,
                    options,
                    model=model[0],
                    subcort_masks=None,
                )
                y = to_categorical(labels, num_classes=2)
                model[1].fit(
                    X,
                    y,
                    batch_size=batch_size,
                    initial_epoch=options["initial_epoch_2"],
                    epochs=options["max_epochs"] + 50,
                    verbose=2,
                    shuffle=True,
                    validation_split=options["train_split"],
                    callbacks=[
                        early_stopping_monitor,
                        model_checkpoint,
                        csv_logger,
                        json_logging_callback,
                    ],
                )
            else:
                pass
        else:
            if os.path.isfile(datapath):
                print(datapath + " exists, loading now")
                # datapath = options['data_path']
                print("====> DNN2 // loading training data from HDF5 dataset")
                X, y, X_val, y_val = load_dataset(datapath, options)
                print("\n\n")

                print("====> DNN2 // fitting model", "\n")
                print_data_shape(X)

                model[1].fit(
                    X,
                    y,
                    batch_size=batch_size,
                    epochs=options["max_epochs_2"],
                    verbose=2,
                    shuffle="batch",
                    validation_data=(X_val, y_val),
                    callbacks=[
                        early_stopping_monitor,
                        model_checkpoint,
                        csv_logger,
                        json_logging_callback,
                    ],
                )
            else:
                X, labels = load_training_data(
                    train_x_data,
                    train_y_data,
                    options,
                    model=model[0],
                    subcort_masks=None,
                )
                y = to_categorical(labels, num_classes=2)
                if options["save_as_hdf5"]:
                    print("\n HDF5 dataset is being created: {}".format(datapath))
                    create_dataset(datapath, X, y)
                    print("\n\n")

                print("====> DNN2 // fitting model", "\n")
                print_data_shape(X)

                model[1].fit(
                    X,
                    y,
                    batch_size=batch_size,
                    epochs=options["max_epochs_2"],
                    verbose=2,
                    shuffle=True,
                    validation_split=options["train_split"],
                    callbacks=[
                        early_stopping_monitor,
                        model_checkpoint,
                        csv_logger,
                        json_logging_callback,
                    ],
                )

            copy_most_recent_model(
                os.path.join(options["weight_paths"], "checkpoints"), net_model
            )
    return model


def nifti2ants(input_np, affine, header):
    nifti = nib.Nifti1Image(input_np, affine=affine, header=header)
    output_ants = ants.convert_nibabel.from_nibabel(nifti)
    return output_ants


def transform_img(
    bidsfilepath,
    bidsfileentities,
    origfilepath,
    transformpath,
    targetspace=None,
    invert=False,
    interpolation="nearestneighbor",
):
    print(f"writing data transformed to the {targetspace} space")
    t = ants.read_transform(transformpath)
    if invert:
        t = t.invert()

    img = ants.image_read(bidsfilepath)
    origimg = ants.image_read(origfilepath)
    img_t = ants.apply_ants_transform_to_image(
        transform=t,
        image=img,
        reference=origimg,
        interpolation=interpolation,
    )
    if targetspace is None:
        outname = bidsfilepath.replace(f"_space-{bidsfileentities['space']}", "")
    else:
        outname = bidsfilepath.replace(bidsfileentities["space"], targetspace)
    img_t.to_filename(outname)


def test_scan(
    model,
    test_x_data,
    options,
    transit=None,
    save_nifti=False,
    uncertainty=True,  # TODO
    candidate_mask=None,
    T=20,
):
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
    batch_size = options["mini_batch_size"]
    # get_scan name and create an empty nii image to store segmentation
    scans = test_x_data.keys()
    flair_scans = [test_x_data[s]["FLAIR"] for s in scans]
    flair_image = load_nii(flair_scans[0]).get_data()
    header = load_nii(flair_scans[0]).header
    affine = header.get_qform()
    seg_image = np.zeros_like(flair_image)
    var_image = np.zeros_like(flair_image)

    # get test paths
    _, scan = os.path.split(flair_scans[0])
    test_folder = os.path.join(options["test_folder"], options["experiment"])
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    # compute lesion segmentation in batches of size options['batch_size']

    for batch, centers in load_test_patches(
        test_x_data,
        options,
        options["patch_size"],
        options["batch_size"],
        options["min_th"],
        candidate_mask,
    ):
        if uncertainty:
            # predict uncertainty
            y_pred, y_pred_var = predict_uncertainty(
                model, batch, batch_size=batch_size, T=T
            )
        else:
            y_pred = model.predict(np.squeeze(batch), batch_size=batch_size, verbose=1)
        [x, y, z] = np.stack(centers, axis=1)
        seg_image[x, y, z] = y_pred[:, 1]
        if uncertainty:
            var_image[x, y, z] = y_pred_var[:, 1]

    if save_nifti:
        # out_scan = nib.Nifti1Image(seg_image, np.eye(4))
        out_scan = nib.Nifti1Image(seg_image, affine=affine, header=header)
        out_scan.to_filename(options["test_mean_name"])

        if uncertainty:
            out_scan = nib.Nifti1Image(var_image, affine=affine, header=header)
            out_scan.to_filename(options["test_var_name"])

    if transit is not None:
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        out_scan = nib.Nifti1Image(seg_image, affine=affine, header=header)
        test_name = str.replace(scan, "_flair.nii.gz", "") + "_out_pred_mean_0.nii.gz"
        out_scan.to_filename(os.path.join(test_folder, test_name))

        if uncertainty:
            out_scan = nib.Nifti1Image(var_image, affine=affine, header=header)
            test_name = (
                str.replace(scan, "_flair.nii.gz", "") + "_out_pred_var_0.nii.gz"
            )
            out_scan.to_filename(os.path.join(test_folder, test_name))

        if not os.path.exists(os.path.join(test_folder, options["experiment"])):
            os.mkdir(os.path.join(test_folder, options["experiment"]))

        out_scan = nib.Nifti1Image(seg_image, affine=affine, header=header)
        test_name = str.replace(scan, "_flair.nii.gz", "") + "_out_pred_0.nii.gz"
        out_scan.to_filename(os.path.join(test_folder, test_name))

    return (seg_image, var_image, header) if uncertainty else (seg_image, None, header)


def copy_most_recent_model(path, net_model):
    files = os.listdir(path)
    paths = [
        os.path.join(path, basename) for basename in files if basename.endswith(".h5")
    ]
    latest_model = max(paths, key=os.path.getctime)
    copyfile(latest_model, os.path.join(path, net_model) + ".h5")


def test_model(
    model,
    test_x_data,
    options,
    performance=False,
    uncertainty=True,
    transforms=None,
    orig_files=None,
    invert_xfrm=True,
):
    outputs = {}
    threshold = options["th_dnn_train_2"]
    scan = options["test_scan"] + "_"

    # scans = test_x_data.keys()
    # flair_scans = [test_x_data[s]["FLAIR"] for s in scans]
    # header = load_nii(flair_scans[0]).header

    # organize experiments
    # first network
    options["test_name"] = os.path.join(
        options["pred_folder"],
        f"{options['fullid']}_space-MNI152NLin2009aSym_acq-{options['experiment']}0_pred.nii.gz",
    )
    options["test_mean_name"] = os.path.join(
        options["pred_folder"],
        f"{options['fullid']}_space-MNI152NLin2009aSym_acq-{options['experiment']}Mean0_pred.nii.gz",
    )
    options["test_var_name"] = os.path.join(
        options["pred_folder"],
        f"{options['fullid']}_space-MNI152NLin2009aSym_acq-{options['experiment']}Var0_pred.nii.gz",
    )
    pred_var_0_img = None
    pred_var_1_img = None
    skip = False

    if os.path.isfile(options["test_mean_name"]):
        pred_mean_0 = nib.load(options["test_mean_name"]).get_data()
        header = nib.load(options["test_mean_name"]).header
        skip = True

    if skip and uncertainty:
        if os.path.isfile(options["test_var_name"]):
            pred_var_0 = nib.load(options["test_var_name"]).get_data()
            header = nib.load(options["test_var_name"]).header
            pred_var_0_img = nifti2ants(
                pred_var_0,
                affine=header.get_qform(),
                header=header,
            )
        else:
            skip = False

    if not skip:
        pred_mean_0, pred_var_0, header = test_scan(
            model[0],
            test_x_data,
            options,
            save_nifti=True,
            uncertainty=uncertainty,
            T=20,
        )

        pred_var_0_img = nifti2ants(
            pred_var_0, affine=header.get_qform(), header=header
        )

    # pred_mean_0_img = nifti2ants(pred_mean_0, affine=None, header=header)
    outputs["pred_mean_0_path"] = options["test_mean_name"]

    if pred_var_0_img is not None:
        outputs["pred_var_0_path"] = options["test_var_name"]

    # pred_mean_0_img = nifti2ants(pred_mean_0, affine=header.get_qform(), header=header)

    # second network
    options["test_name"] = os.path.join(
        options["pred_folder"],
        f"{options['fullid']}_space-MNI152NLin2009aSym_acq-{options['experiment']}1_pred.nii.gz",
    )
    options["test_mean_name"] = os.path.join(
        options["pred_folder"],
        f"{options['fullid']}_space-MNI152NLin2009aSym_acq-{options['experiment']}Mean1_pred.nii.gz",
    )
    options["test_var_name"] = os.path.join(
        options["pred_folder"],
        f"{options['fullid']}_space-MNI152NLin2009aSym_acq-{options['experiment']}Var1_pred.nii.gz",
    )

    skip = False

    if os.path.isfile(options["test_mean_name"]):
        pred_mean_1 = nib.load(options["test_mean_name"]).get_data()
        skip = True

    if skip and uncertainty:
        if os.path.isfile(options["test_var_name"]):
            pred_var_1 = nib.load(options["test_var_name"]).get_data()
            header = nib.load(options["test_var_name"]).header
            pred_var_1_img = nifti2ants(
                pred_var_1,
                affine=None,
                header=header,
            )
        else:
            skip = False

    if not skip:
        pred_mean_1, pred_var_1, header = test_scan(
            model[1],
            test_x_data,
            options,
            save_nifti=True,
            uncertainty=uncertainty,
            T=50,
            candidate_mask=pred_mean_0 > threshold,
        )

        pred_var_1_img = nifti2ants(pred_var_1, affine=None, header=header)

    # pred_mean_1_img = nifti2ants(pred_mean_1, affine=header.get_qform(), header=header)
    # pred_mean_1_img = nifti2ants(pred_mean_1, affine=None, header=header)
    # outputs['pred_mean_1_img'] = pred_mean_1_img
    outputs["pred_mean_1_path"] = options["test_mean_name"]

    if pred_var_1_img is not None:
        outputs["pred_var_1_path"] = options["test_var_name"]

    if performance:
        # postprocess the output segmentation
        print("postprocessing")
        # options["test_name"] = options["experiment"] + "_out_CNN.nii.gz"
        maskpath, labelpath, _ = post_processing(
            pred_mean_1, options, header, save_nifti=True
        )

        outputs["mask_path"] = maskpath
        outputs["label_path"] = labelpath

        # outputs = [pred_mean_0, pred_var_0, pred_mean_1, pred_var_0 out_segmentation, lpred, count]

    return outputs
