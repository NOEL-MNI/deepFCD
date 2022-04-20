options = {}

# options['experiment'] = 'exp_LoSo_holdout_MNI'
options['load_checkpoint'] = False
options['cross_validation'] = False
options['train_1'] = True
options['train_2'] = True

options['subcorticalMask'] = False

options['modalities'] = ['T1', 'FLAIR']
options['x_names'] = ['_t1.nii.gz', '_flair.nii.gz']

options['y_names'] = ['_lesion_bin.nii.gz']
options['submask_names'] = ['subcorticalMask_final_negative.nii.gz']
options['batch_size'] = 250000 # 200704 # (512x392)
options['net_verbose'] = 11
options['seed'] = 666

options['nb_classes'] = 2
options['channels'] = 2
options['base_filters'] = 48
options['activation'] = 'relu'
options['optimizer'] = 'Adadelta'

options['dropout_mc'] = True
options['dropout_1'] = 0.2
options['dropout_2'] = 0.2
options['dropout_3'] = 0.4

# continue interrupted training sessions
options['load_checkpoint_1'] = False
options['load_checkpoint_2'] = False
options['continue_training_2'] = False
if options['continue_training_2']:
    options['initial_epoch_2'] = 35
else:
    options['initial_epoch_2'] = 1

# cases to exclude
options['exclude'] = ['.DS_Store', '._.DS_Store', '078', '095']

# threshold to select voxels for training, discarding CSF and darker WM in FLAIR
options['thr'] = 0.1
options['min_th'] = options['thr'] # z-scored [10%ile=0.15, 15%ile=0.28, 20%ile=0.38, 25%ile=0.46, 28%ile=0.5, 5%ile=-0.05, 0%ile=-6]
options['th_dnn_train_2'] = 0.1 # probabilistic

# post-processing binary threshold. After segmentation, probabilistic masks are binarized using a defined threshold.
options['t_bin'] = 0.1
# The resulting binary mask is filtered by removing lesion regions with lesion size before a defined value
options['l_min'] = 25
options['patch_size'] = (16,16,16)
options['train_split'] = 0.25

# maximum number of epochs used to train the model
options['max_epochs'] = 60
options['max_epochs_1'] = 60
options['max_epochs_2'] = 100
options['patience'] = 15
# file paths to store the network parameter weights. These can be reused for posterior use.
options['weight_paths'] = None
# randomize training features before fitting the model.
options['randomize_train'] = True
