import cv2
import copy
import io
import PIL
import os
import h5py
import numpy as np
import cfg as CFG
from yolo_uav import *
import pdb

def brightness_augment(image):
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype=np.float64)
    random_bright = 0.5 + np.random.uniform()
    image[:, :, 2] = image[:, :, 2]*random_bright
    image[:, :, 2][image[:, :, 2] > 255]  = 255
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def read_train_batch(data_images, data_boxes):  
    idx = np.random.choice(data_images.shape[0], replace=False)
    batch_image = data_images[idx]
    batch_boxes = data_boxes[idx]
    batch_boxes = batch_boxes.reshape((-1, 5))    
    image = PIL.Image.open(io.BytesIO(batch_image))
    image_data = np.array(image, dtype=np.float)           
    return np.array(image_data), np.array(batch_boxes)

def augment_image(image_data, bboxes, model_width, model_height, jitter=False):
    h, w, c = image_data.shape
    if jitter:        
        scale = np.random.uniform() / 10. + 1.
        image_data = cv2.resize(image_data, (0,0), fx = scale, fy = scale)    
        ## translate the image
        max_offx = (scale-1.) * w
        max_offy = (scale-1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        image_data = image_data[offy : (offy + h), offx : (offx + w)]        
        flip = np.random.binomial(1, .5)
        if flip > 0.5: 
            image_data = cv2.flip(image_data, 1)
            
    image_data = cv2.resize(image_data, (model_height, model_width))
    for bbox in bboxes:
        for attr in (1,3): # adjust xmin and xmax
            if jitter: bbox[attr] = int(bbox[attr] * scale - offx)
            bbox[attr] = int(bbox[attr] * float(model_width) / w)     
            bbox[attr] = max(min(bbox[attr], model_width), 0)
        for attr in 2,4: # adjust ymin and ymax
            if jitter: bbox[attr] = int(bbox[attr] * scale - offy)
            bbox[attr] = int(bbox[attr] * float(model_height) / h)
            bbox[attr] = max(min(bbox[attr], model_height), 0)
        if jitter and flip > 0.5:
            xmin = bbox[1]
            bbox[1] = model_width - bbox[3]
            bbox[3] = model_width - xmin  
    return image_data, bboxes

class DataBatchGenerator:
    def __init__(self, hdf5_data=None,
                       model_w=CFG.IMAGE_WIDTH, 
                       model_h=CFG.IMAGE_HEIGHT, 
                       batch_size = CFG.BATCH_SIZE,
                       anchors_path = CFG.ANCHORS_PATH,
                       train='train',
                       jitter=False):
        self.model_w = model_w
        self.model_h = model_h
        self.batch_size = batch_size
        self.jitter = jitter
        self.hdf5 = hdf5_data
        self.training_instances  = self.hdf5[train+ '/images'].shape[0]
        self.H5_IMAGES    = self.hdf5[train+ '/images']
        self.H5_BOXES   = self.hdf5[train+ '/boxes']
        self.anchors = get_anchors(anchors_path)
        if CFG.SHALLOW_DETECTOR:
            self.anchors = self.anchors * 2
            
    def flow_from_hdf5(self):
        while True:
            x_batch = np.zeros((self.batch_size, CFG.IMAGE_HEIGHT, CFG.IMAGE_WIDTH, 3))
            y_batch = np.zeros((self.batch_size, CFG.FEAT_H, CFG.FEAT_W, CFG.N_ANCHORS, 5 + CFG.N_CLASSES))
        
            for i in range(self.batch_size):    
                # import pdb; pdb.set_trace()    
                image_data, bboxes = read_train_batch(self.H5_IMAGES, self.H5_BOXES)
                image_data, bboxes = augment_image(image_data, bboxes, self.model_w, self.model_h, self.jitter)

                orig_size = np.array([image_data.shape[1], image_data.shape[0]])
                orig_size = np.expand_dims(orig_size, axis=0)
                # normalize the image data 
                image_data /= 255.

                x_batch[i] = image_data
                
                boxes = bboxes.reshape((-1, 5))
                boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
                boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
                boxes_xy = boxes_xy / orig_size
                boxes_wh = boxes_wh / orig_size
                boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)
                for box in boxes:
                    label = int(box[-1])
                    one_hot = np.eye(CFG.N_CLASSES)[label]
                    xc, yc, w, h = box[0:4]
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
                    
                    for j in range(len(self.anchors)):
                        anchor_bb = BoundBox(0, 0, self.anchors[i][0], self.anchors[i][1]) 
                        iou    = bbox_iou(shifted_box, anchor_bb)
                        if max_iou < iou:
                            best_anchor = j
                            max_iou     = iou
                    # print(c,r, best_anchor, max_iou, object_mask)
                    # pdb.set_trace()
                    if r < CFG.FEAT_W and c < CFG.FEAT_H:
                        y_batch[i, c, r, best_anchor, :] = object_mask    # Construct Feature map ground truth
                        # y_batch[i, c, r, :, :] = CFG.N_ANCHORS* [object_mask]    # Construct Feature map ground truth
                        
            y_batch = y_batch.reshape([self.batch_size,  CFG.FEAT_H, CFG.FEAT_W, CFG.N_ANCHORS*(5 + CFG.N_CLASSES)])
            yield x_batch, y_batch
    