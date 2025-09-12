from keras import backend as K
from keras.layers import Activation, Conv3D, Dropout, Flatten, Input, MaxPooling3D
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# from keras.layers.advanced_activations import PReLU, LeakyReLU

K.set_image_dim_ordering("th")
import warnings

warnings.filterwarnings("ignore")


def off_the_shelf(input, options):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # base_filters = 48
    base_filters = options["base_filters"]

    c1 = Conv3D(
        base_filters, (3, 3, 3), border_mode="same", activation=options["activation"]
    )(input)
    # c1 = Conv3D(base_filters, (3, 3, 3), border_mode='same')(input)
    # p1 = LeakyReLU()(c1)
    b1 = BatchNormalization(axis=channel_axis)(c1)
    if options["dropout_mc"]:
        b1 = Dropout(options["dropout_1"])(b1)
    m1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(b1)

    c2 = Conv3D(
        base_filters * 2,
        (3, 3, 3),
        border_mode="same",
        activation=options["activation"],
    )(m1)
    # c2 = Conv3D(base_filters*2, (3, 3, 3), border_mode='same')(m1)
    # p2 = LeakyReLU()(c2)
    b2 = BatchNormalization(axis=channel_axis)(c2)
    if options["dropout_mc"]:
        b2 = Dropout(options["dropout_2"])(b2)
    m2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(b2)

    return m2


def create_off_the_shelf(options):
    """
    Creates a custom off-the-shelf CNN
    :param nb_classes: number of classes
    :return: Keras Model with 1 input (patch_size) and 1 output
    """
    nb_classes = options["nb_classes"]
    channels = options["channels"]
    shape = options["patch_size"]

    if K.image_dim_ordering() == "th":
        init = Input((channels, shape[0], shape[1], shape[2]))
    else:
        init = Input((shape[0], shape[1], shape[2], channels))

    x = off_the_shelf(init, options)

    # Dropout
    x = Dropout(options["dropout_3"])(x)

    x = Conv3D(nb_classes, (3, 3, 3), border_mode="same")(x)
    x = MaxPooling3D((4, 4, 4))(x)

    x = Flatten()(x)

    # Output
    out = Activation("softmax")(x)

    model = Model(init, output=[out], name="off_the_shelf")

    return model
