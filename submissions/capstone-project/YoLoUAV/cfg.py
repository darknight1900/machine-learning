'''
Main configuration file for YOLOv2 Project.
Modify every time one would like to train on a new dataset
'''
import numpy as np

# If a feature extractor performed 5 max-pooling --> Image resolution being reduced 2^5 = 32 times
# Most current state-of-the-art models have max-pooling layers (August, 2017)
# FEATURE_EXTRACTOR = 'Darknet19'
FEATURE_EXTRACTOR = 'MobileNet'
N_CLASSES         = 2
N_ANCHORS         = 5

BATCH_SIZE        = 4

IMAGE_HEIGHT      = 608
IMAGE_WIDTH       = 608
INPUT_SHAPE       = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

SHALLOW_DETECTOR        = True
USE_THREE_SCALE_FEATURE = True

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

if SHALLOW_DETECTOR:
    SHRINK_FACTOR     = 16
else:
    SHRINK_FACTOR     = 32

FEAT_H = IMAGE_HEIGHT // SHRINK_FACTOR
FEAT_W = IMAGE_WIDTH // SHRINK_FACTOR

MAX_DETECTION_PER_IMAGE = 10
CLASSES_PATH='model_data/uav_classes.txt'
ANCHORS_PATH='model_data/uav_anchors.txt'


###############################################################################
### Note: if SHALLOW_DETECTOR True, we need scale up the anchor boxes
###############################################################################
