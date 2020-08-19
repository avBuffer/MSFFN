#! /usr/bin/env python
# coding=utf-8
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class YoloTest(object):
    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE

        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))

        self.weight_file = cfg.TEST.WEIGHT_FILE
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD

        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path = cfg.TEST.ANNOT_PATH

        self.ground_truth_path = cfg.TEST.GROUND_TRUTH_PATH
        if os.path.exists(self.ground_truth_path):
            shutil.rmtree(self.ground_truth_path)
        os.makedirs(self.ground_truth_path)

        self.predicted_path = cfg.TEST.PREDICTED_PATH
        if os.path.exists(self.predicted_path):
            shutil.rmtree(self.predicted_path)
        os.makedirs(self.predicted_path) 
      
        self.write_image = cfg.TEST.WRITE_IMAGE        
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        if os.path.exists(self.write_image_path):
            shutil.rmtree(self.write_image_path)
        os.makedirs(self.write_image_path)

        self.show_label = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.lwir_input_data = tf.placeholder(dtype=tf.float32, name='lwir_input_data')
            self.trainable = tf.placeholder(dtype=tf.bool, name='trainable')

        model = YOLOV3(self.input_data, self.lwir_input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)


    def predict(self, image, lwir_image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        lwir_image_data = utils.image_preporcess(lwir_image, [self.input_size, self.input_size])
        lwir_image_data = lwir_image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run([self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={self.input_data: image_data, self.lwir_input_data: lwir_image_data, self.trainable: False})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        return bboxes


    def evaluate(self):
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()                
                image_file = annotation[0]
                image_name = image_file.split('/')[-1]
                image = cv2.imread(image_file)

                lwir_image_file = annotation[1]
                lwir_image = cv2.imread(lwir_image_file)

                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[2:]])
                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                
                ground_truth_file = os.path.join(self.ground_truth_path, image_name.replace('.jpg', '.txt'))
                num_bbox_gt = len(bboxes_gt)
                print('=> ground truth of %s' % image_name, 'ground_truth_file %s' % ground_truth_file, 
                      'bbox_gt.len %d' % num_bbox_gt)
                
                with open(ground_truth_file, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                
                predict_result_file = os.path.join(self.predicted_path, image_name.replace('.jpg', '.txt'))
                bboxes_pr = self.predict(image)
                print('=> predict result of %s:' % image_name, 'predict_result_file %s' % predict_result_file, 
                      'bboxes_pr.len %d' % len(bboxes_pr))

                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(os.path.join(self.write_image_path, image_name), image)
                    lwir_image = utils.draw_bbox(lwir_image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(os.path.join(self.write_image_path, image_name.replace('.jpg', '_lwir.jpg')), lwir_image)
   
                with open(predict_result_file, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())


if __name__ == '__main__':
    YoloTest().evaluate()
