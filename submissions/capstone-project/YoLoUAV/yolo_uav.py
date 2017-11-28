import os
import cv2
import numpy as np
import PIL
import io
import h5py

import tensorflow as tf
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input, Conv2D, Reshape

from feature_extractor import MobileNetFeature, Darknet19Feature
import cfg as CFG
import pdb
import keras.backend as K
from keras.regularizers import l2
from keras.layers.merge import concatenate


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    x1_min = box1.x - box1.w / 2
    x1_max = box1.x + box1.w / 2
    y1_min = box1.y - box1.h / 2
    y1_max = box1.y + box1.h / 2

    x2_min = box2.x - box2.w / 2
    x2_max = box2.x + box2.w / 2
    y2_min = box2.y - box2.h / 2
    y2_max = box2.y + box2.h / 2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h
    union = box1.w * box1.h + box2.w * box2.h - intersect
    return float(intersect) / union


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readlines()
            try:
                anchors = [anchor.rstrip().split(',') for anchor in anchors]
                anchors = sum(anchors, [])
                anchors = [float(x) for x in anchors]
            except:
                anchors = CFG.YOLO_ANCHORS
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return CFG.YOLO_ANCHORS


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


class BoundBox:
    def __init__(self, x, y, w, h, c=None, classes=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

    def to_array(self):
        return np.array([self.x, self.y, self.w, self.h, int(self.get_label())])


def get_recall_precision(bboxes_pred, bboxes_gt, iou_threshold=0.4, num_classes=CFG.N_CLASSES):
    classes = get_classes(CFG.CLASSES_PATH)
    classes_detection_results = dict.fromkeys(
        range(num_classes))  # label_id : TP, FP, FN
    for key, value in classes_detection_results.items():
        classes_detection_results[key] = [0, 0, 0]  # TP, FP, FN
    # calculate TP (true positive) and FP (False Positive)
    for bp in bboxes_pred:
        for i in range(len(bboxes_gt)):
            bg = bboxes_gt[i]
            iou_matched = bbox_iou(bp, bg) > iou_threshold
            label_matched = (bp.get_label() == bg.get_label())
            # pdb.set_trace()
            if iou_matched and label_matched and bg.c > 0.0 and bp.c > 0.0:
                classes_detection_results[bp.get_label()][0] += 1
                bg.c = 0.0
                bp.c = 0.0
                break
        if bp.c > 0.0:
            classes_detection_results[bp.get_label()][1] += 1
    # pdb.set_trace()
    for bg in bboxes_gt:
        if bg.c > 0.0:
            classes_detection_results[bg.get_label()][2] += 1
    # Now calculate Precision and Recall for each classes
    for key, value in classes_detection_results.items():
        prec = value[0] / float(value[0] + value[1] + 0.00001)
        recall = value[0] / float(value[0] + value[2] + 0.00001)
        print(classes[key], 'True Positive:', value[0],
              'False Positive:', value[1], 'False Negative:', value[2])
        print(classes[key], 'precision:', prec, 'recall:', recall)


def compute_recall_precision(hdf5_data, yolo_detector, weights=None, train='valid', num_samples=1024):
    detect_model = yolo_detector.model
    # detect_model.summary()
    assert(os.path.exists(weights))
    detect_model.load_weights(weights)

    total_samples = hdf5_data[train + '/images'].shape[0]
    sample_list = np.random.choice(total_samples, num_samples, replace=False)
    hdf5_images = hdf5_data[train + '/images']
    hdf5_boxes = hdf5_data[train + '/boxes']

    x_batch = np.zeros((num_samples, CFG.IMAGE_HEIGHT, CFG.IMAGE_WIDTH, 3))
    y_batch = []
    cur_id = 0
    for sample_id in sample_list:
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        image = PIL.Image.open(io.BytesIO(hdf5_images[sample_id]))
        orig_size = np.array([image.width, image.height])
        orig_size = np.expand_dims(orig_size, axis=0)

        image = image.resize(
            (CFG.IMAGE_WIDTH, CFG.IMAGE_HEIGHT), PIL.Image.BICUBIC)
        image_data = np.array(image, dtype=np.float)
        image_data /= 255.
        x_batch[cur_id] = image_data

        boxes = hdf5_boxes[sample_id]
        boxes = boxes.reshape((-1, 5))

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
        boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
        boxes_xy = boxes_xy / orig_size
        boxes_wh = boxes_wh / orig_size
        boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)

        for box in boxes:
            label = int(box[-1])
            one_hot = np.eye(CFG.N_CLASSES)[label]
            xc, yc, w, h = box[0:4]
            y_batch.append(BoundBox(xc, yc, w, h, c=1.0, classes=one_hot))

    netouts = detect_model.predict(x_batch, batch_size=CFG.BATCH_SIZE)
    netouts = netouts.reshape(-1, CFG.FEAT_H, CFG.FEAT_W,
                              CFG.N_ANCHORS, (5 + CFG.N_CLASSES))
    y_pred = []
    for i in range(len(netouts)):
        image_data = x_batch[i]
        boxes = yolo_detector.decode_netout(netouts[i])
        y_pred += boxes
    # pdb.set_trace()
    get_recall_precision(y_pred, y_batch)


def draw_boxes(image, boxes, labels):
    for box in boxes:
        xmin = int((box.x - box.w / 2) * image.shape[1])
        xmax = int((box.x + box.w / 2) * image.shape[1])
        ymin = int((box.y - box.h / 2) * image.shape[0])
        ymax = int((box.y + box.h / 2) * image.shape[0])
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, image.shape[1])
        ymax = min(ymax, image.shape[0])

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image.shape[0],
                    (0, 0, 255), 2)
    return image


class YOLODetector(object):
    def __init__(self, feature_extractor_name,
                 image_shape=CFG.INPUT_SHAPE,
                 classes_path=CFG.CLASSES_PATH,
                 anchors_path=CFG.ANCHORS_PATH,
                 shallow_feature=CFG.SHALLOW_DETECTOR,
                 use_three_scale_feature=CFG.USE_THREE_SCALE_FEATURE):
        self.input_shape = image_shape
        self.anchors = get_anchors(anchors_path)
        self.classes = get_classes(classes_path)

        self.nb_class = len(self.classes)
        self.nb_box = len(self.anchors)
        self.batch_size = CFG.BATCH_SIZE

        if CFG.SHALLOW_DETECTOR:
           self.anchors = self.anchors * 2

        assert(self.nb_class == CFG.N_CLASSES)
        assert(self.nb_box == CFG.N_ANCHORS)

        self.output_tensor = None
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.debug = True
        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image = Input(
            shape=(self.input_shape[0], self.input_shape[1], 3))

        if feature_extractor_name == 'Darknet19':
            self.feature_net = Darknet19Feature(input_image,
                                                shallow_detection=shallow_feature,
                                                three_scale_detection=use_three_scale_feature)
        elif feature_extractor_name == 'MobileNet':
            self.feature_net = MobileNetFeature(input_image,
                                                shallow_detection=shallow_feature,
                                                three_scale_detection=use_three_scale_feature)
        else:
            raise Exception(
                'Feature_extractor issupported! Please select from [MobileNet, Darknet19]')

        print(feature_extractor_name + ' output shape is ',
              self.feature_net.output_shape())
        self.grid_h, self.grid_w = self.feature_net.output_shape()

        feature_model = self.feature_net.get_feature_model()

        # create the final object detection layer
        output_tensor = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                               (1, 1), strides=(1, 1),
                               padding='same',
                               name='conv_final',
                               kernel_regularizer=l2(5e-4))(feature_model.output)  # Xavier normal initializer

        # output_tensor = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output_tensor)
        # output = Lambda(lambda args: args[0])([output, self.true_boxes])

        # self.model = Model([input_image, self.true_boxes], output)
        self.model = Model(inputs=feature_model.inputs, outputs=output_tensor)
        # print a summary of the whole model
        # self.model.summary()

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def post_process(self, n_classes=2, anchors=None, iou_threshold=0.5, score_threshold=0.6, mode=2):
        prediction = self.model.output
        N_ANCHORS = 5
        ANCHORS = CFG.YOLO_ANCHORS

        pred_shape = tf.shape(prediction)
        GRID_H, GRID_W = pred_shape[1], pred_shape[2]

        prediction = K.reshape(
            prediction, [-1, pred_shape[1], pred_shape[2], N_ANCHORS, n_classes + 5])

        cx = tf.cast((K.arange(0, stop=GRID_W)), dtype=tf.float32)
        cx = K.tile(cx, [GRID_H])
        cx = K.reshape(cx, [-1, GRID_H, GRID_W, 1])

        cy = K.cast((K.arange(0, stop=GRID_H)), dtype=tf.float32)
        cy = K.reshape(cy, [-1, 1])
        cy = K.tile(cy, [1, GRID_W])
        cy = K.reshape(cy, [-1])
        cy = K.reshape(cy, [-1, GRID_H, GRID_W, 1])

        c_xy = tf.stack([cx, cy], -1)
        c_xy = tf.to_float(c_xy)

        anchors_tensor = tf.to_float(
            K.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2]))
        netout_size = tf.to_float(K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]))

        box_xy = K.sigmoid(prediction[..., :2])
        box_wh = K.exp(prediction[..., 2:4])
        box_confidence = K.sigmoid(prediction[..., 4:5])
        box_class_probs = prediction[..., 5:]

        # Shift center points to its grid cell accordingly (Ref: YOLO-9000 loss function)
        box_xy = (box_xy + c_xy) / netout_size
        box_wh = (box_wh * anchors_tensor) / netout_size
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        # Y1, X1, Y2, X2
        boxes = K.concatenate([box_mins[..., 1:2], box_mins[..., 0:1],  # Y1 X1
                               box_maxes[..., 1:2], box_maxes[..., 0:1]])  # Y2 X2

        box_scores = box_confidence * K.softmax(box_class_probs)
        box_classes = K.argmax(box_scores, -1)

        box_class_scores = K.max(box_scores, -1)
        prediction_mask = (box_class_scores >= score_threshold)

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        # Scale boxes back to original image shape.
        height, width = CFG.IMAGE_HEIGHT, CFG.IMAGE_WIDTH
        image_dims = tf.cast(
            K.stack([height, width, height, width]), tf.float32)
        image_dims = K.reshape(image_dims, [1, 4])
        boxes = boxes * image_dims

        nms_index = tf.image.non_max_suppression(
            boxes, scores, 10, iou_threshold)
        boxes = tf.gather(boxes, nms_index)
        scores = tf.gather(scores, nms_index)
        classes = tf.gather(classes, nms_index)
        return boxes, classes, scores

    def decode_netout(self, netout, obj_threshold=0.6, nms_threshold=0.3):
        grid_h, grid_w, nb_box = netout.shape[:3]
        boxes = []
        # decode the output by the network
        netout[..., 4] = self.sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * \
            self.softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > obj_threshold
        # pdb.set_trace()

        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    # pdb.set_trace()
                    classes = netout[row, col, b, 5:]

                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row, col, b, :4]
                        # center position, unit: image width
                        x = (col + self.sigmoid(x)) / grid_w
                        # center position, unit: image height
                        y = (row + self.sigmoid(y)) / grid_h
                        # pdb.set_trace()
                        w = self.anchors[b][0] * \
                            np.exp(w) / grid_w  # unit: image width
                        # unit: image height
                        h = self.anchors[b][1] * np.exp(h) / grid_h
                        w = min(1.0, w)
                        h = min(1.0, h)
                        confidence = netout[row, col, b, 4]
                        box = BoundBox(x, y, w, h, confidence, classes)
                        boxes.append(box)
                        # pdb.set_trace()
        # suppress non-maximal boxes
        for c in range(self.nb_class):
            sorted_indices = list(
                reversed(np.argsort([box.classes[c] for box in boxes])))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0:
                    continue
                else:
                    for j in range(i + 1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                            boxes[index_j].classes[c] = 0
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.get_score() > obj_threshold]
        return boxes

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t

        e_x = np.exp(x)

        return e_x / e_x.sum(axis, keepdims=True)
