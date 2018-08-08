#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2018/8/8 15:18
# @File    : Eval_FM.py

import tensorflow as tf
import numpy as np
import argparse
import os
from time import time


def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='../data/old/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ctr',
                        help='Choose a dataset.')
    return parser.parse_args()


class LoadTestData(object):
    def __init__(self, path, dataset):
        self.path = path + dataset + "/"
        self.trainfile_path = self.path + "train/"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features = {}
        # self.features_M = self.map_features()
        self.Test_Data, self.Y_ = self.read_data(self.testfile)

    def map_features(self):
        files = sorted(os.listdir(self.trainfile_path))
        for file in files:
            self.read_features(self.trainfile_path+file)
        # self.read_features(self.testfile)
        self.read_features(self.validationfile)
        print("features_M:", len(self.features))
        return len(self.features)

    def read_features(self, file):
        with open(file) as f:
            line = f.readline()
            i = len(self.features)
            while line:
                items = line.split("\t")[1].split(",")
                for item in items:
                    if item not in self.features:
                        self.features[item] = i
                        i = i + 1
                line = f.readline()

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open(file)
        X_ = []
        Y_ = []
        line = f.readline()
        while line:
            items = line.strip().split("\t")
            X_.append([int(item) for item in items[1].split(",")])
            line = f.readline()
            Y_.append(0)
        f.close()
        print("len(x): "+str(len(X_)))
        return X_, Y_


graph = tf.Graph()
t1 = time()
with graph.as_default():
    args = parse_args()
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        ckpt_file = "F:\\attentional_factorization_machine\\pretrain\\fm_ctr_10\\ctr_10"

        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
        saver.restore(sess, ckpt_file)

        # Load the TestSet data
        DATA = LoadTestData(args.path, args.dataset)

        # Load tensors
        feature_embeddings = graph.get_tensor_by_name('feature_embeddings:0')
        nonzero_embeddings = graph.get_tensor_by_name('nonzero_embeddings:0')
        feature_bias = graph.get_tensor_by_name('feature_bias:0')
        bias = graph.get_tensor_by_name('bias:0')
        fm = graph.get_tensor_by_name('fm:0')
        fm_out = graph.get_tensor_by_name('fm_out:0')
        out = graph.get_tensor_by_name('out:0')
        prediction = graph.get_tensor_by_name('prediction:0')
        train_features = graph.get_tensor_by_name('train_features_fm:0')
        train_labels = graph.get_tensor_by_name('train_labels_fm:0')
        dropout_keep = graph.get_tensor_by_name('dropout_keep_fm:0')
        train_phase = graph.get_tensor_by_name('train_phase_fm:0')

        num_samples = len(DATA.Test_Data)
        print("num_sampels: "+str(num_samples))
        X = np.asarray(DATA.Test_Data)
        Y = np.reshape(np.asarray(DATA.Y_), (len(DATA.Y_), 1))
        feed_dict = {train_features: X, train_labels: Y, dropout_keep: 1.0, train_phase: False}
        predictions = sess.run(prediction, feed_dict=feed_dict)

        y_pred = np.reshape(predictions, (num_samples,))

        print("shape: " + str(y_pred.shape))

        with open(args.path+args.dataset+"/predict.txt", 'w') as f_pre:
            for i in y_pred:
                f_pre.write(str(i)+"\n")

        print("Cost %.2f s" % (time()-t1))
