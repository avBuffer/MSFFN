#! /usr/bin/env python
# coding=utf-8
import os
import sys
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':	
	pb_file = 'pbs/Pedestrian_yolov3_loss-99.2099_162.8037.ckpt-8.pb'
    out_path = 'data/out'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

	is_image = False
	if is_image:
		test_file = 'data/imgs/visible.jpg'
		lwir_test_file = 'data/imgs/lwir.jpg'
    else:
        test_file = 'data/videos/visible.mp4' # for camera id: 0
        lwir_test_file = 'data/videos/lwir.mp4' # for camera id: 1

    num_classes = 1
    input_size = 416
    score_thresh = 0.3
    iou_thresh = 0.45

    graph = tf.Graph()
    return_elements = ['input/input_data:0', 'input/lwir_input_data:0', 
                       'pred_sbbox/concat_2:0', 'pred_mbbox/concat_2:0', 'pred_lbbox/concat_2:0']
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.Session(graph=graph) as sess:
        if is_image:
            original_image = cv2.imread(test_file)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]

            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            lwir_original_image = cv2.imread(lwir_test_file)
            lwir_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            lwir_image_data = utils.image_preporcess(np.copy(lwir_original_image), [input_size, input_size])
            lwir_image_data = lwir_image_data[np.newaxis, ...]

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[2], return_tensors[3], return_tensors[4]],
                                                           feed_dict={return_tensors[0]: image_data, return_tensors[1]: lwir_image_data})
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, score_thresh)
            bboxes = utils.nms(bboxes, iou_thresh, method='nms')

            image = utils.draw_bbox(original_image, bboxes)
            image = Image.fromarray(image)
            visible_file = os.path.join(out_path, 'visible_result.jpg')
            image.save(visible_file)

            lwir_image = utils.draw_bbox(lwir_original_image, bboxes)
            lwir_image = Image.fromarray(lwir_image)
            lwir_file = os.path.join(out_path, 'lwir_result.jpg')
            lwir_image.save(lwir_file)

        else: 
            vid = cv2.VideoCapture(test_file)
            lwir_vid = cv2.VideoCapture(lwir_test_file)
            
            idx = 0
            while True:
                ret, frame = vid.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    print('No visible image!')
                    break

                lwir_ret, lwir_frame = lwir_vid.read()
                if lwir_ret:
                    lwir_frame = cv2.cvtColor(lwir_frame, cv2.COLOR_BGR2RGB)
                    vimage = Image.fromarray(lwir_frame)
                else:
                    print('No lwir image!')
                    break

                frame_size = frame.shape[:2]
                image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]
                
                lwir_image_data = utils.image_preporcess(np.copy(lwir_frame), [input_size, input_size])
                lwir_image_data = lwir_image_data[np.newaxis, ...]

                prev_time = time.time()
                pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[2], return_tensors[3], return_tensors[4]],
                                                               feed_dict={return_tensors[0]: image_data, return_tensors[1]: lwir_image_data})
                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, score_thresh)
                bboxes = utils.nms(bboxes, iou_thresh, method='nms')
                
                image = utils.draw_bbox(frame, bboxes)
                lwir_image = utils.draw_bbox(lwir_frame, bboxes)
                
                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = 'time: %.2f ms' % (1000 * exec_time)

                result = np.asarray(image)
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                visible_file = os.path.join(out_path, '%s_visible.jpg' % str(idx))
                cv2.imwrite(visible_file, result)

                lwir_result = np.asarray(lwir_image)                
                lwir_result = cv2.cvtColor(lwir_image, cv2.COLOR_RGB2BGR)
                lwir_file = os.path.join(out_path, '%s_lwir.jpg' % str(idx))
                cv2.imwrite(lwir_file, lwir_result)

                print('idx=', idx, 'visible_file=', visible_file, 'lwir_file=', lwir_file, info)
                idx += 1
                
                cv2.namedWindow('visible', cv2.WINDOW_NORMAL)
                cv2.imshow('visible', result)
                #cv2.namedWindow('lwir', cv2.WINDOW_NORMAL)
                #cv2.imshow('lwir', lwir_result)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
