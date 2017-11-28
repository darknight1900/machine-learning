"""
DarKNet19 Keras Implementation:
YOLO9000: Better, Faster, Stronger
https://arxiv.org/pdf/1612.08242
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D
from keras.layers import BatchNormalization, Activation
from keras.layers import GlobalAvgPool2D
from keras.layers.advanced_activations import LeakyReLU

def Darknet19(image_tensor=None, num_classes=1000, include_top=False):
    """
    DarkNet-19 Architecture Definition
    Parameters
    ----------
    image_tensor: tensor
        Input tensor. Default: None
    num_classes: int
        Number of classes for classfication tasks. Default: 1000
    include_top: bool
        Whether includes the last layer (only needs for classfication tasks). Default: False
    """
    if image_tensor is None:
        image_tensor = Input(shape=(None, None, 3))

    x = conv_block(image_tensor, 32, (3, 3))  # << --- Input layer
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 64, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 128, (3, 3))
    x = conv_block(x, 64, (1, 1))
    x = conv_block(x, 128, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 256, (3, 3))
    x = conv_block(x, 128, (1, 1))
    x = conv_block(x, 256, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 512, (3, 3))
    x = conv_block(x, 256, (1, 1))
    x = conv_block(x, 512, (3, 3))
    x = conv_block(x, 256, (1, 1))
    x = conv_block(x, 512, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 1024, (3, 3))
    x = conv_block(x, 512, (1, 1))
    x = conv_block(x, 1024, (3, 3))
    x = conv_block(x, 512, (1, 1))
    x = conv_block(x, 1024, (3, 3))    # ---> feature extraction ends here

    if include_top:
        x = Conv2D(num_classes, (1, 1), activation='linear', padding='same')(x)
        x = GlobalAvgPool2D()(x)
        x = Activation(activation='softmax')(x)

    darknet = Model(image_tensor, x)

    return darknet


def conv_block(x, filters, kernel_size, name=None):
    """
    Standard YOLOv2 Convolutional Block as suggested in YOLO9000 paper
    :param x:
    :param filters:
    :param kernel_size:
    :param kernel_regularizer:
    :return:
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
               use_bias=False, name=name)(x)
    x = BatchNormalization(name=name if name is None else 'batch_norm_%s' % name)(x)
    x = LeakyReLU(alpha=0.1, name=name if name is None else 'leaky_relu_%s' % name)(x)
    return x
