"""
Overfit one image with 1000 epochs to test the loss function properly
"""
import random
import h5py
import os
import PIL
import io
import cv2
import time
import datetime

from argparse import ArgumentParser
from loss import custom_loss
from yolo_uav import *
import numpy as np
import cfg as CFG

import keras
import tensorflow as tf
from datagen import DataBatchGenerator
from keras.models import load_model
from keras.optimizers import Adam, RMSprop

import keras.backend as K
import matplotlib.pyplot as plt

parser = ArgumentParser(
    description="Retrain the Yolo-UAV for a dataset")

parser.add_argument('-p',
                    '--data_path',
                    help='path to HDF5 file containing dataset',
                    default='~/data/PascalVOC/VOCdevkit/pascal_voc_07_12_person_vehicle.hdf5')


parser.add_argument('-w',
                    '--weights_path',
                    help="Path to pre-trained weight files",
                    type=str, default=None)

parser.add_argument('-e',
                    '--num_epochs',
                    help='Number of epochs for training',
                    type=int, default=100)

parser.add_argument('-b',
                    '--batch_size',
                    help='Number of batch size',
                    type=int, default=CFG.BATCH_SIZE)

parser.add_argument('-lr',
                    '--learning_rate',
                    help='Learning Rate',
                    type=float, default=1e-5)

def get_current_timestamp():
    ts = time.time()
    cur_time = datetime.datetime.fromtimestamp(
        ts).strftime('%Y-%m-%d-%H:%M:%S')
    return cur_time

def _main_():
    args = parser.parse_args()
    data_path = args.data_path
    weights_path = args.weights_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    # ###################
    # PREPARE DATA INPUT
    # ###################

    anchors = get_anchors(CFG.ANCHORS_PATH)
    classes = get_classes(CFG.CLASSES_PATH)
    data_path = os.path.expanduser(data_path)

    if CFG.SHALLOW_DETECTOR:
        anchors = anchors * 2
    assert(CFG.N_ANCHORS == len(anchors))
    assert(CFG.N_CLASSES == len(classes))
    assert(os.path.exists(data_path))
    hdf5_data = h5py.File(data_path, 'r')
    num_training = hdf5_data['train/images'].shape[0]

    print("==========================")
    print('\t anchors:', anchors)
    print('\t classes:', classes)
    print('\t train_path:', data_path)
    print('\t num_training:', num_training)
    print("==========================")

    yolo_detector = YOLODetector(feature_extractor_name=CFG.FEATURE_EXTRACTOR)
    detect_model = yolo_detector.model
    detect_model.summary()
    if weights_path and os.path.exists(weights_path):
        detect_model.load_weights(weights_path)
    # #################
    # COMPILE AND RUN
    # #################


    train_batch_gen = DataBatchGenerator(hdf5_data, train='train', jitter=True)
    valid_batch_gen = DataBatchGenerator(hdf5_data, train='valid')

    logging = TensorBoard()
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    train_steps_per_epoch = train_batch_gen.training_instances // batch_size
    valid_steps_per_epoch = valid_batch_gen.training_instances // batch_size
    print('train_steps_per_epoch=', train_steps_per_epoch)
    print('valid_steps_per_epoch=', valid_steps_per_epoch)

    num_loop_epochs = 5
    loop = num_epochs // num_loop_epochs
    for i in range(loop):
        optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-09, decay=1e-08)
        detect_model.compile(optimizer=optimizer, loss=custom_loss)

        cur_time = get_current_timestamp()
        weight_name = 'weights/' + 'best_{}{}{}_loop_{}_{}.h5'.format(
            CFG.FEATURE_EXTRACTOR, int(CFG.SHALLOW_DETECTOR),
            int(CFG.USE_THREE_SCALE_FEATURE), i, cur_time)

        checkpoint = ModelCheckpoint(
            weight_name, monitor='val_loss', save_weights_only=True, save_best_only=True)
        detect_model.fit_generator(generator=train_batch_gen.flow_from_hdf5(),
                                   validation_data=valid_batch_gen.flow_from_hdf5(),
                                   steps_per_epoch=train_steps_per_epoch,
                                   validation_steps=valid_steps_per_epoch,
                                   callbacks=[checkpoint, logging],
                                   epochs=num_loop_epochs,
                                   workers=1,
                                   verbose=1)
        print('Complete traing for loop {}, saved weights {}'.format(i, weight_name))
        prec_rec = compute_recall_precision(
            hdf5_data, yolo_detector, weight_name, train='valid', num_samples=1024)

        weight_name = 'weights/' + '{}{}{}_lp{}_{}_{}_{}_{}_{}.h5'.format(
            CFG.FEATURE_EXTRACTOR, int(CFG.SHALLOW_DETECTOR),
            int(CFG.USE_THREE_SCALE_FEATURE), i, get_current_timestamp(), 
            prec_rec[0][1], prec_rec[0][2],prec_rec[1][1], prec_rec[1][2])
        detect_model.save_weights(weight_name)
        
if __name__ == "__main__":
    _main_()
