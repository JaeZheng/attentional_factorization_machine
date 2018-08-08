#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2018/8/6 10:15
# @File    : demo.py

each_num = 3000000
num = 9
file_len = 0
with open("../data/ctr/ctr.train.libfm", 'r') as f:
    file_len = len(f.readlines())
    print("file_len: " + str(file_len))

with open("../data/ctr/ctr.train.libfm", 'r') as f:
    count = 0
    iter = 0
    content = ""
    f_out = []
    for i in range(num):
        f_tmp = open("../data/ctr/train/ctr.train.libfm_%d"%(i), 'w')
        f_out.append(f_tmp)
    for line in f:
        if iter<num:
            f_out[iter].write(line)
        else:
            f_out[iter-1].write(line)
        count += 1
        if count % 1000000 == 0:
            print("Count: " + str(count))
        if count == each_num:
            count = 0
            iter += 1
            if iter<num:
               f_out[iter-1].close()
        elif iter==num and count==file_len%each_num:
            f_out[iter-1].close()




