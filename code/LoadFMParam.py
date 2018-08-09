#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2018/8/8 14:10
# @File    : LoadFMParam.py

import tensorflow as tf
import numpy as np

graph = tf.Graph()
with graph.as_default():

    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        ckpt_file = "F:\\attentional_factorization_machine\\pretrain\\fm_ctr_10\\ctr_10"
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
        saver.restore(sess, ckpt_file)

        feature_embeddings = graph.get_tensor_by_name('feature_embeddings:0')
        feature_bias = graph.get_tensor_by_name('feature_bias:0')
        bias = graph.get_tensor_by_name('bias:0')

        saver.restore(sess, ckpt_file)
        fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])

        np.savetxt("C:\\Users\\Jae\\Desktop\\fe.txt", fe)
        np.savetxt("C:\\Users\\Jae\\Desktop\\fb.txt", fb)
        print("shape of fe: " + str(fe.shape))
        print("shape of fb: " + str(fb.shape))
        print("type of b: " +str(type(b)))
        print(b)
