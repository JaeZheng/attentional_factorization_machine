#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2018/8/6 10:15
# @File    : demo.py

from LoadCTRData import LoadCTRData as DATA

each_num = 3000000
num = 9

with open("../data/ctr/ctr.train.libfm", 'r') as f:
    count = 0
    iter = 0
    content = ""
    f_out = []
    for i in range(num):
        f_tmp = open("../data/ctr/ctr.train.libfm_%d"%(i), 'w')
        f_out.append(f_tmp)
    for line in f:
         f_out[iter].write(line)
         count += 1
         if count % 100000 == 0:
             print("Count: " + str(count))
         if count == each_num:
             f_out[iter].close()
             iter += 1
             count = 0




