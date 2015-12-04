# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:29:17 2015

@author: dennis
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""




'''
NEXT: EXTRACT PATCHES AROUND DETECTIONS
GET BETTER DATA
'''


import os
import tensorflow as tf
import time
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import network as net
from read_detection_data import read_data_sets
from preprocess import preprocess
from augment_images import augment_data_sets
from validate_detection import validate_detection

#import input_data


                        
def get_batch(data, labels, index, size):
    indices = range(1,data.shape[0],size)
    #print indices
    if index >= len(indices) - 1:
        return data[indices[index]:], labels[indices[index]:]
    return data[indices[index]:indices[index+1]], labels[indices[index]:indices[index+1]]
    


    
pretrained = True
projdir = '/home/dennis/CellClassification/'
modelname = 'wayback'
modelfolder = projdir + 'nets/' + modelname + '/'
datadir = projdir+'data/'
resultsdir = projdir + 'validationresults/' + modelname + '/'
train_dir = datadir+'working_folder/training'
label_dir = datadir+'working_folder/labels'
if not os.path.exists(modelfolder):
    os.mkdir(modelfolder)
if not os.path.exists(resultsdir):
    os.mkdir(resultsdir)
    
sess = tf.Session()

# Here comes the net

x = tf.placeholder("float", shape=[None, None, None, 3])
y_ = tf.placeholder("float", shape=[None, None, None, 1])
xsize = tf.placeholder(tf.int32, shape=[2])
y_conv = net.network(x,xsize,modelname)

diff = tf.sub(y_,y_conv)
eucl_loss = tf.reduce_sum(tf.mul(diff,diff))
train_step = tf.train.AdamOptimizer(1e-3).minimize(eucl_loss)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if os.path.isfile(modelfolder+'checkpoint') and pretrained:
    saver.restore(sess, tf.train.latest_checkpoint(modelfolder))

######### READ DATA
det_data = read_data_sets(train_dir, label_dir)
traindata = preprocess(det_data.traindata)
valdata = preprocess(det_data.valdata)
numtrain = traindata.shape[0]
numval = valdata.shape[0]
#valdata = det_data.valdata
trainlabels = det_data.trainlabels * 100
vallabels = det_data.vallabels * 100

maxinner = 15
maxouter = 1000

#batchindex = 0
batchsize = 50
for outer in range(maxouter):
    #print 'outer', outer

    ######### AUGMENT IMAGES
    print 'Augmenting...'
    augtrain, auglabel = augment_data_sets(traindata, trainlabels)
    #augtrain, auglabel = (traindata, trainlabels)

    ######## VISUALISATION AND VALIDATION
    if outer%1== 0 and outer > -1:
        results = []  
        for j in range(numval/10):     #image = mnist.test.images[i]
            res = sess.run([eucl_loss, y_conv], feed_dict={x: valdata[j:j+1], y_: vallabels[j:j+1], xsize: [det_data.traindata.shape[1],det_data.traindata.shape[2]]})  
            results.append(res[0])
            validate_detection(res[1], det_data.valdata[j,:,:,:], j, resultsdir)
        print outer, np.mean(results)    
    
    ######### Training
    print 'Training...'
    for inner in range(maxinner): 
        #print 'inner', inner
        
        ######## SHUFFLE HERE
        shuffle = np.arange(numtrain)
        np.random.shuffle(shuffle)
        train = augtrain[shuffle]
        labels = auglabel[shuffle]

        ######## TRAINING
        for batchindex in range(numtrain/batchsize):
            trainbatch, labelsbatch = get_batch(train, labels, batchindex, 10)
            sess.run(train_step,feed_dict={x: trainbatch, y_: labelsbatch, xsize: [train.shape[1],train.shape[2]]})
            
    saver.save(sess, modelfolder+'model.ckpt', global_step=outer)

sess.close()