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
NEXT: ADD CLASSIFICATION
GET BETTER DATA
'''


import os
import tensorflow as tf
import numpy as np

from network import network
from preprocess import preprocess

from detection import Detection
from classification import Classification
                        
def get_batch(data, labels, index, size):
    indices = range(1,data.shape[0],size)
    #print indices
    if index >= len(indices) - 1:
        return data[indices[index]:], labels[indices[index]:]
    return data[indices[index]:indices[index+1]], labels[indices[index]:indices[index+1]]
    
def get_task(taskname):
    if taskname == 'detection':
        return Detection()
    elif taskname == 'classification':
        return Classification()
    else:
        print 'Task "' + taskname + '" not supported.'
        return

def train_tfnet(task, projdir, modelname, sessionname, dataset, pretrained, maxouter=1000, maxinner=15, batchsize=50, step=1e-3):
    
    # FOLDER VARIABLES
    sessiondir = projdir + 'nets/' + modelname + '_' + sessionname + '/'
    resultsdir = projdir + 'validationresults/' + modelname + '_' + sessionname + '/'
    datadir = projdir + 'data/' + dataset
    train_dir = datadir+'/training'
    label_dir = datadir+'/labels'
    if not os.path.exists(sessiondir):
        os.mkdir(sessiondir)
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
        
    sess = tf.Session()
    
    # NETWORK INIT
    x = tf.placeholder("float", shape=[None, None, None, 3])    
    xsize = tf.placeholder(tf.int32, shape=[2])
    y_conv = network(x,xsize,modelname)

    taskobj = get_task(task)
    y_, loss = taskobj.loss(y_conv)

    train_step = tf.train.AdamOptimizer(step).minimize(loss)
    #correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    
    # SAVER
    saver = tf.train.Saver()
    if os.path.isfile(sessiondir+'checkpoint') and pretrained:
        saver.restore(sess, tf.train.latest_checkpoint(sessiondir))
    
    ######### READ DATA
    det_data = taskobj.read_training_sets(train_dir, label_dir)
    traindata = preprocess(det_data.traindata)
    valdata = preprocess(det_data.valdata)
    numtrain = traindata.shape[0]
    numval = valdata.shape[0]
    #valdata = det_data.valdata
    trainlabels = det_data.trainlabels
    vallabels = det_data.vallabels
    
    # TRAINING
    for outer in range(maxouter):
        #print 'outer', outer      
    
        ######## VISUALISATION AND VALIDATION
        if outer%1== 0 and outer > -1:
            results = []  
            for j in range(numval/10):     #image = mnist.test.images[i]
                res = sess.run([loss, y_conv], feed_dict={x: valdata[j:j+1], y_: vallabels[j:j+1], xsize: [valdata.shape[1],valdata.shape[2]]})  
                results.append(res[0])
                taskobj.validate(res[1], det_data.valdata[j,:,:,:], j, resultsdir)
            print outer, np.mean(results)    
        
        ######### Training
        
        ######### AUGMENT IMAGES
        print 'Augmenting...'
        augtrain, auglabel = taskobj.augment_images(traindata, trainlabels)
        #augtrain, auglabel = (traindata, trainlabels)
        
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
                trainbatch, labelsbatch = get_batch(train, labels, batchindex, batchsize)
                sess.run(train_step,feed_dict={x: trainbatch, y_: labelsbatch, xsize: [train.shape[1],train.shape[2]]})
                
        saver.save(sess, sessiondir+'model.ckpt', global_step=outer)
    
    sess.close()
    
def test_tfnet(task, projdir, modelname, sessionname, dataset, patchflag=False, patchsize=100):

    # FOLDER VARIABLES
    sessiondir = projdir + 'nets/' + modelname + '_' + sessionname + '/'
    resultsdir = projdir + 'testresults/' + modelname + '_' + sessionname + '/'
    datadir = projdir + 'data/' + dataset
    test_dir = datadir+'/testing'
    
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
    
    # NETWORK INIT
    x = tf.placeholder("float", shape=[None, None, None, 3])
    xsize = tf.placeholder(tf.int32, shape=[2])
    y_conv = network(x,xsize,modelname)
    
    taskobj = get_task(task)    
    
    sess = tf.Session()
    
    saver = tf.train.Saver()
    if os.path.isfile(sessiondir+'checkpoint'):
        saver.restore(sess, tf.train.latest_checkpoint(sessiondir))
    else:
        print 'Model not pretrained'
    
    # DATA INIT
    det_data = taskobj.read_testing_sets(test_dir)
    testdata = preprocess(det_data.testdata)
    
    # TESTING
    for j in range(testdata.shape[0]):
        print j
        res = sess.run([y_conv], feed_dict={x: testdata[j:j+1], xsize: [testdata.shape[1],testdata.shape[2]]})
        taskobj.validate(res[0], det_data.testdata[j,:,:,:], j, resultsdir, patchflag, patchsize)