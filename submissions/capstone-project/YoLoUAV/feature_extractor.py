import os

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Reshape, Activation, Conv2D, Input
from keras.layers import MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet
from keras.layers.advanced_activations import LeakyReLU

from keras_darknet19 import Darknet19
from keras_mobilenet import depthwise_conv_block, relu6


def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


def space_to_depth_x4(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=4."""
    # Import currently required to make Lambda work.
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=4)


def space_to_depth_x4_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=4.
    """
    return (input_shape[0], input_shape[1] // 4, input_shape[2] // 4, 16 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    16 * input_shape[3])


class FeatureExtractor(object):
    """Abstract class for feature extracor
    """
    # to be defined in each subclass

    def __init__(self, input_tensor):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")

    def output_shape(self):
        return self.feature_model.get_output_shape_at(-1)[1:3]

    def get_feature_model(self):
        return self.feature_model


class Darknet19Feature(FeatureExtractor):
    """Original YoLov2 with Darknet19 as feature extractor 
    Parameters
    ----------
    input_tensor : tensor
        Input tensor with shape (height, width, num_channel)
    weights:
        Load pretrained weights with COCO dataset
    shallow_detection : bool
        Whether use a shallow net. In the original YoLov2 design, the size of the last 
        feature is 1/32 of input due to 5 maxpooling. When shallow_detection is enabled, 
        we only keep all the layes upto the first 4 maxpooling. This is to hopefully improve
        performance for small objects. 
    three_scale_detection : bool
        Whether to use 3 scale of features for detection. The original YoLov2 will use feature 
        from last layer and one earlier feature and concatenate them togther. We extend this idea
        by introducing an extra scale of feature to improve the detection accuracy. 
    """

    def __init__(self, input_tensor, weights=None, shallow_detection=True, three_scale_detection=False):

        fine_grained_layers = [17, 27, 43]  # [1/4, 1/8, 1/16]
        if shallow_detection:
            fine_grained_layers = fine_grained_layers[0:2]
            num_fina_layers = 512
            final_feature_layer = 43  # Total 44 layer
        else:
            fine_grained_layers = fine_grained_layers[1:]
            num_fina_layers = 1024
            final_feature_layer = -1  # total 75 layers

        feature_model = Darknet19(input_tensor, include_top=False)
        feature_model = Model(inputs=feature_model.input,
                              outputs=feature_model.layers[final_feature_layer].output)

        if weights == 'COCO':
            print("Loading trained COCO weights...")
            model_path = os.path.join('weights', 'yolo-coco-m.h5')
            trained_model = load_model(model_path)
            trained_layers = trained_model.layers
            feature_layers = feature_model.layers
            for i in range(0, min(len(feature_layers), len(trained_layers))):
                weights = trained_layers[i].get_weights()
                feature_layers[i].set_weights(weights)

        x0 = feature_model.layers[fine_grained_layers[0]].output
        x1 = feature_model.layers[fine_grained_layers[1]].output
        x2 = feature_model.output

        if shallow_detection:
            x0 = Conv2D(8, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x0)
            x1 = Conv2D(32, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x1)
            num_fina_layers = 512

        else:
            x0 = Conv2D(16, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x0)
            x1 = Conv2D(64, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x1)
            num_fina_layers = 1024

        # Layer 19
        x2 = Conv2D(num_fina_layers, (3, 3), strides=(1, 1),
                    padding='same', name='conv_19', use_bias=False)(x2)
        x2 = BatchNormalization(name='norm_19')(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)

        # Layer 20
        x2 = Conv2D(num_fina_layers, (3, 3), strides=(1, 1),
                    padding='same', name='conv_20', use_bias=False)(x2)
        x2 = BatchNormalization(name='norm_20')(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)

        # earlier net feature
        x0 = BatchNormalization(name='norm_space_to_depth_x4')(x0)
        x0 = LeakyReLU(alpha=0.1)(x0)
        x0_reshaped = Lambda(
            space_to_depth_x4,
            output_shape=space_to_depth_x4_output_shape,
            name='space_to_depth_x4')(x0)

        # earlier net feature
        x1 = BatchNormalization(name='norm_space_to_depth_x2')(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)
        x1_reshaped = Lambda(
            space_to_depth_x2,
            output_shape=space_to_depth_x2_output_shape,
            name='space_to_depth_x2')(x1)

        if three_scale_detection:
            x = concatenate([x0_reshaped, x1_reshaped, x2])
        else:
            x = concatenate([x1_reshaped, x2])

        x = Conv2D(num_fina_layers, (3, 3), strides=(1, 1),
                   padding='same', name='conv_detection', use_bias=False)(x)
        x = BatchNormalization(name='norm_detection_feature')(x)
        x = LeakyReLU(alpha=0.1)(x)
        self.feature_model = Model(feature_model.inputs, x)

    def normalize(self, image):
        return image / 255.


class MobileNetFeature(FeatureExtractor):
    """MobileNet based YoLo 
    Parameters
    ----------
    input_shape : tensor
        Input tensor shape (height, width, num_channel)
    weights: string
        Load pretrained weights with imagenet dataset
    shallow_detection : bool
        Whether use a shallow net. In the original YoLov2 design, the size of the last 
        feature is 1/32 of input due to 5 maxpooling. When shallow_detection is enabled, 
        we only keep all the layes upto the first 4 maxpooling. This is to hopefully improve
        performance for small objects. 
    three_scale_detection : bool
        Whether to use 3 scale of features for detection. The original YoLov2 will use feature 
        from last layer and one earlier feature and concatenate them togther. We extend this idea
        by introducing an extra scale of feature to improve the detection accuracy. 
    """

    def __init__(self, input_tensor, weights='imagenet', shallow_detection=False, three_scale_detection=False):

        fine_grained_layers = [21, 33, 69]  # [1/4, 1/8, 1/16]

        if shallow_detection:
            fine_grained_layers = fine_grained_layers[0:2]
            final_feature_layer = 69
        else:
            fine_grained_layers = fine_grained_layers[1:]
            final_feature_layer = -1

        feature_model = MobileNet(
            input_tensor=input_tensor, include_top=False, weights=None)
        feature_model = Model(inputs=feature_model.input,
                              outputs=feature_model.layers[final_feature_layer].output)

        if weights == 'imagenet':
            print('Loading pretrained weights from ImageNet...')
            trained_model = MobileNet(input_shape=(
                224, 224, 3), include_top=False, weights=weights)
            trained_layers = trained_model.layers
            feature_layers = feature_model.layers
            for i in range(0, min(len(feature_layers), len(trained_layers))):
                weights = trained_layers[i].get_weights()
                feature_layers[i].set_weights(weights)

        x0 = feature_model.layers[fine_grained_layers[0]].output
        x1 = feature_model.layers[fine_grained_layers[1]].output
        x2 = feature_model.output

        if shallow_detection:
            x0 = Conv2D(8, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x0)
            x1 = Conv2D(32, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x1)
            num_final_layers = 512

        else:
            x0 = Conv2D(16, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x0)
            x1 = Conv2D(64, (1, 1), strides=(1, 1),
                        padding='same', use_bias=False)(x1)
            num_final_layers = 1024

        x2 = depthwise_conv_block(x2, num_final_layers, 1.0, block_id=14)
        x2 = depthwise_conv_block(x2, num_final_layers, 1.0, block_id=15)

        x1 = BatchNormalization()(x1)
        x1 = Lambda(relu6)(x1)
        x1_reshaped = Lambda(
            space_to_depth_x2,
            output_shape=space_to_depth_x2_output_shape,
            name='space_to_depth_x2')(x1)

        x0 = BatchNormalization()(x0)
        x0 = Lambda(relu6)(x0)
        x0_reshaped = Lambda(
            space_to_depth_x4,
            output_shape=space_to_depth_x4_output_shape,
            name='space_to_depth_x4')(x0)

        if three_scale_detection:
            x = concatenate([x0_reshaped, x1_reshaped, x2])
        else:
            x = concatenate([x1_reshaped, x2])
        x = depthwise_conv_block(x, num_final_layers, 1.0, block_id=16)
        self.feature_model = Model(feature_model.inputs, x)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        return image
