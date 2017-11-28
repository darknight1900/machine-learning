"""
Overfit one image with 1000 epochs to test the loss function properly
"""
import random
import h5py
import os
import PIL
import io
import cv2

from argparse import ArgumentParser
from loss import custom_loss
from yolo_uav import *
import numpy as np
import cfg as CFG

import keras
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt

parser = ArgumentParser(description="Over-fit one sample to validate YOLOv2 Loss Function")

parser.add_argument('-p', '--path', help="Path to training text file ",
                    type=str, default=None)

parser.add_argument('-w', '--weights', help="Path to pre-trained weight files",
                    type=str, default=None)

parser.add_argument('-e', '--epochs', help='Number of epochs for training',
                    type=int, default=1000)

parser.add_argument('-b', '--batch', help='Number of batch size',
                    type=int, default=1)

parser.add_argument(
    '-d',
    '--data_path',
    help='path to HDF5 file containing pascal voc dataset',
    default='~/data/PascalVOC/VOCdevkit/pascal_voc_07_12_person_vehicle.hdf5')

args = parser.parse_args()
annotation_path = args.path
WEIGHTS_FILE    = args.weights
BATCH_SIZE      = args.batch
EPOCHS          = args.epochs

def _main_():
    # ###################
    # PREPARE DATA INPUT
    # ###################
    anchors = get_anchors(CFG.ANCHORS_PATH)
    classes = get_classes(CFG.CLASSES_PATH)

    if CFG.SHALLOW_DETECTOR:
        anchors = anchors * 2
    print(anchors)
    test_size = 128
    voc_path = os.path.expanduser(args.data_path)
    voc = h5py.File(voc_path, 'r')
    total_test_instances = voc['train/images'].shape[0]

    test_list = np.random.choice(total_test_instances, test_size, replace=False)

    x_batch = np.zeros((test_size, CFG.IMAGE_HEIGHT, CFG.IMAGE_WIDTH, 3))
    y_batch = np.zeros((test_size, CFG.FEAT_H, CFG.FEAT_W, CFG.N_ANCHORS, 5 + CFG.N_CLASSES))
    b_batch = []

    cur_id = 0
    for test_id in sorted(test_list):
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        image = PIL.Image.open(io.BytesIO(voc['train/images'][test_id]))
        orig_size = np.array([image.width, image.height])
        orig_size = np.expand_dims(orig_size, axis=0)

        image = image.resize((CFG.IMAGE_WIDTH, CFG.IMAGE_HEIGHT), PIL.Image.BICUBIC)
        image_data = np.array(image, dtype=np.float)
        image_data /= 255.
        x_batch[cur_id] = image_data

        boxes = voc['train/boxes'][test_id]
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
            b_batch.append(BoundBox(xc, yc, w, h, c=1.0, classes=one_hot))

            object_mask = np.concatenate([[xc, yc, w, h], [1.0], one_hot])  # A cell in grid map`

            center_x = xc * CFG.FEAT_W
            center_y = yc * CFG.FEAT_H
            r = int(np.floor(center_x))
            c = int(np.floor(center_y))
            fw = w * CFG.FEAT_W
            fh = h * CFG.FEAT_H
            
            # find the anchor that best predicts this box
            best_anchor = -1
            max_iou     = -1
            shifted_box = BoundBox(0, 0, fw, fh)
            
            for i in range(len(anchors)):
                anchor_bb = BoundBox(0, 0, anchors[i][0], anchors[i][1]) 
                iou    = bbox_iou(shifted_box, anchor_bb)
                if max_iou < iou:
                    best_anchor = i
                    max_iou     = iou
            print(c,r, best_anchor, max_iou, object_mask)
            if r < CFG.FEAT_W and c < CFG.FEAT_H:
                y_batch[cur_id, c, r, best_anchor, :] = object_mask    # Construct Feature map ground truth
            
        cur_id += 1

    # pdb.set_trace()
    y_batch = y_batch.reshape([test_size,  CFG.FEAT_H, CFG.FEAT_W, CFG.N_ANCHORS*(5 + CFG.N_CLASSES)])

    yolo_detector = YOLODetector(feature_extractor_name=CFG.FEATURE_EXTRACTOR)    
    # yolo_detector.model.summary()
    # #################
    # COMPILE AND RUN
    # #################
    yolo_detector.model.compile(optimizer='adam', loss=custom_loss)
    
    num_steps = 200
        
    yolo_detector.model.fit(x_batch, y_batch, batch_size=CFG.BATCH_SIZE, epochs=num_steps)
    yolo_detector.model.save_weights('overfit.weights')

    netout = yolo_detector.model.predict(x_batch, batch_size=CFG.BATCH_SIZE)
    netouts = netout.reshape(-1, CFG.FEAT_H, CFG.FEAT_W, CFG.N_ANCHORS, (5 + CFG.N_CLASSES))
    idx = 0
    boxes_pred = []
    for i in range(len(netouts)):
        image_data = x_batch[i]
        boxes  = yolo_detector.decode_netout(netouts[i])
        boxes_pred += boxes
        img = draw_boxes(image_data, boxes, classes)
        img = np.array(img * 255, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/tmp/output/img_'+str(i)+'.jpg',img)    
    # import pdb; pdb.set_trace()
    get_recall_precision(boxes_pred, b_batch)
if __name__ == "__main__":
    _main_()