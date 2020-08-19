#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg
cfg = __C


# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.NET_TYPE = 'mobilenetv2' # 'darknet53' 'mobilenetv2'
__C.YOLO.CLASSES = "data/classes/pedestrian.names"
__C.YOLO.ANCHORS = "data/anchors/basline_anchors.txt"

__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5

__C.YOLO.UPSAMPLE_METHOD = "resize"
__C.YOLO.FUSION_METHOD = "add"


# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "D:/datasets/Pedestrians/pedestrian_train.txt"
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG = False

__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_END = 1e-8
__C.TRAIN.WARMUP_EPOCHS = 5

__C.TRAIN.FISRT_STAGE_EPOCHS = 20
__C.TRAIN.SECOND_STAGE_EPOCHS = 50

__C.TRAIN.INITIAL_WEIGHT = "Pedestrian_yolov3_demo.ckpt"
__C.TRAIN.CKPT_PATH = "ckpts"


# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "D:/datasets/Pedestrians/pedestrian_val.txt"
__C.TEST.BATCH_SIZE = 1
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False

__C.TEST.WEIGHT_FILE = "ckpts/Pedestrian_yolov3_loss-99.2099_162.8037.ckpt-8"
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45

__C.TEST.GROUND_TRUTH_PATH = "mAP/ground-truth/"
__C.TEST.PREDICTED_PATH = "mAP/predicted/"

__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = "mAP/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True

__C.TEST.SHOW_LABEL = True
