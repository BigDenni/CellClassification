# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:01:53 2015

@author: dennis
"""

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from network import mynet
from read_detection_data import read_data_sets_testing
from preprocess import preprocess
                        
    
test_dir = '/home/dennis/Documents/Cell_Detection_Classification/data/working_folder/testing'

x = tf.placeholder("float", shape=[None, None, None, 3])
y_ = tf.placeholder("float", shape=[None, None, None, 1])
xsize = tf.placeholder(tf.int32, shape=[2])
y_conv = mynet(x,xsize)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(eucl_loss)

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('/home/dennis/Documents/Cell_Detection_Classification/nets'))

det_data = read_data_sets_testing(test_dir)
testdata = preprocess(det_data.testdata)

for j in range(testdata.shape[0]):
    print j
    testimage = testdata[j]
    res = sess.run([y_conv], feed_dict={x: testdata[j:j+1], xsize: [testdata.shape[1],testdata.shape[2]]})
    image = res[0]
    image = image[0,:,:,0]
    
    mpimg.imsave('/home/dennis/Documents/Cell_Detection_Classification/data/testresults/'+str(j)+'_outfile.jpg', image)
    mpimg.imsave('/home/dennis/Documents/Cell_Detection_Classification/data/testresults/'+str(j)+'_original.jpg', det_data.testdata[j,:,:,:])
    