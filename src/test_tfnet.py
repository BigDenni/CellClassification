# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:01:53 2015

@author: dennis
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import network as net
from read_detection_data import read_data_sets_testing
from preprocess import preprocess
from validate_detection import validate_detection
                        

projdir = '/home/dennis/CellClassification/'
modelname = 'wayback'
modelfolder = projdir + 'nets/' + modelname + '/'
datadir = projdir+'data/'
resultsdir = projdir + 'testresults/' + modelname + '/'
test_dir = datadir+'working_folder/testing'

if not os.path.exists(resultsdir):
    os.mkdir(resultsdir)

x = tf.placeholder("float", shape=[None, None, None, 3])
y_ = tf.placeholder("float", shape=[None, None, None, 1])
xsize = tf.placeholder(tf.int32, shape=[2])
y_conv = net.network(x,xsize,modelname)

sess = tf.Session()

saver = tf.train.Saver()
if os.path.isfile(modelfolder+'checkpoint'):
    saver.restore(sess, tf.train.latest_checkpoint(modelfolder))
else:
    print 'Model not pretrained'

det_data = read_data_sets_testing(test_dir)
testdata = preprocess(det_data.testdata)

for j in range(testdata.shape[0]):
    print j
    testimage = testdata[j]
    res = sess.run([y_conv], feed_dict={x: testdata[j:j+1], xsize: [testdata.shape[1],testdata.shape[2]]})
    validate_detection(res[0], det_data.testdata[j,:,:,:], j, resultsdir)
    