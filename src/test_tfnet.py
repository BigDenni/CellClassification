# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:01:53 2015

@author: dennis
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from network import mynet2
from read_detection_data import read_data_sets_testing
from preprocess import preprocess
                        

projdir = '/home/dennis/CellClassification/'
modelname = 'Little'
modelfolder = projdir + 'nets/' + modelname + '/'
datadir = projdir+'data/'
test_dir = datadir+'working_folder/testing'

x = tf.placeholder("float", shape=[None, None, None, 3])
y_ = tf.placeholder("float", shape=[None, None, None, 1])
xsize = tf.placeholder(tf.int32, shape=[2])
y_conv = mynet2(x,xsize)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(eucl_loss)

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
    image = res[0]
    image = image[0,:,:,0]
    image_t = (image > 20) * 1
    
    mpimg.imsave(datadir+'testresults/'+str(j)+'_outfile.jpg', image)
    mpimg.imsave(datadir+'testresults/'+str(j)+'_outans.jpg', image)
    mpimg.imsave(datadir+'testresults/'+str(j)+'_original.jpg', det_data.testdata[j,:,:,:])
    