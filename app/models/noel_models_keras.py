from models.model_builder import *
from keras.utils import multi_gpu_model
from keras.optimizers import Adadelta
from keras import losses


def off_the_shelf_model(options):
    """
    Inputs:
    - model_options:

    Output:
    - nets = list of models (CNN1, CNN2)
    """

    # model options

    # --------------------------------------------------
    # first model
    # --------------------------------------------------
    model_1 = create_off_the_shelf(options)
    if options['parallel_gpu']:
        model_1 = multi_gpu_model(model_1, gpus=2)
    model_1.compile(optimizer=Adadelta(), loss=losses.binary_crossentropy, metrics=['accuracy'])

    # --------------------------------------------------
    # second model
    # --------------------------------------------------
    model_2 = create_off_the_shelf(options)
    if options['parallel_gpu']:
        model_2 = multi_gpu_model(model_2, gpus=2)
    model_2.compile(optimizer=Adadelta(), loss=losses.binary_crossentropy, metrics=['accuracy'])

    return [model_1, model_2]
