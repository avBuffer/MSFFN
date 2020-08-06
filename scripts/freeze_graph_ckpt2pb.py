#! /usr/bin/env python
# coding=utf-8
import os
import sys
import tensorflow as tf
from core.yolov3 import YOLOV3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':    
    ckpt_file = 'ckpts/Pedestrian_yolov3_loss-99.2099_162.8037.ckpt-8'
    pb_file = 'pbs/Pedestrian_yolov3_loss-99.2099_162.8037.ckpt-8.pb'
    output_node_names = ['input/input_data', 'input/lwir_input_data', 
                         'pred_sbbox/concat_2', 'pred_mbbox/concat_2', 'pred_lbbox/concat_2']

    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')
        lwir_input_data = tf.placeholder(dtype=tf.float32, name='lwir_input_data')

    model = YOLOV3(input_data, lwir_input_data, trainable=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)
    
    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=output_node_names)
    with tf.gfile.GFile(pb_file, 'wb') as f:
        f.write(converted_graph_def.SerializeToString())
