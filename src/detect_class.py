# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:39:47 2015

@author: dennis
"""

import os
import tensorflow as tf
import numpy as np

from network import network
from preprocess import preprocess

from detection import Detection
#from classification import Classification
import detection
import matplotlib.pyplot as plt
import scipy

def detect_class(projdir, detmodel, classmodel, sessionname, dataset, detargs, classargs):
    
    ##### Only testing needed
    
    # FOLDER VARIABLES
    detsessiondir = projdir + 'nets/' + detmodel + '_' + sessionname + '/'
    clssessiondir = projdir + 'nets/' + classmodel + '_' + sessionname + '/'
    resultsdir = projdir + 'testresults/' + detmodel + '_' + classmodel +  '_' + sessionname + '/'
    datadir = projdir + 'data/' + dataset
    test_dir = datadir+'/testing'
    
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
        
    
    # Detmodel init
    #
    detgraph = tf.Graph()
    with detgraph.as_default():
        detsess = tf.Session()
        detx = tf.placeholder("float", shape=[None, None, None, 3])
        #xsize = tf.placeholder(tf.int32, shape=[2])
        dety = network(detx,detmodel, detargs)
        
        detsaver = tf.train.Saver()
        if os.path.isfile(detsessiondir+'checkpoint'):
            detsaver.restore(detsess, tf.train.latest_checkpoint(detsessiondir))
        else:
            print 'Detection model not pretrained'
    
    
    clsgraph = tf.Graph()
    with clsgraph.as_default():
        clssess = tf.Session()
        
        # Classmodel init
        clsx = tf.placeholder("float", shape=[None, None, None, 3])
        #xsize = tf.placeholder(tf.int32, shape=[2])
        clsy = network(clsx, classmodel, classargs)    
        clssaver = tf.train.Saver()
        if os.path.isfile(clssessiondir+'checkpoint'):
            clssaver.restore(clssess, tf.train.latest_checkpoint(clssessiondir))
        else:
            print 'Classification model not pretrained'
         
    
    # Restore models with saver
    
    
    
    detobj = Detection()
    #clsobj = Classification()
    
    # Read test data
    det_data, filenames = detobj.read_testing_sets(test_dir)
    testdata = preprocess(det_data.testdata)
    
    #init classifications structure
    print 'GO'
    # for each test image:
    for j in range(testdata.shape[0]):
        print j
        classifications = {}
    
        for i in range(classargs['nouts']):
            classifications[i] = ([],[])
        #imagename = str(j) + '.jpg'        
        
        # run detection net on test image
        with detgraph.as_default():
            detres = detsess.run([dety], feed_dict={detx: testdata[j:j+1]})
        image = detres[0][0,:,:,0]
        image = scipy.ndimage.gaussian_filter(image, sigma=(1, 1), order=0)
        #fig = plt.figure(frameon=False)
        #ax = fig.add_subplot(111)
        #ax.imshow(image, aspect='normal')
        #fig.savefig(resultsdir+str(j)+'_detout.jpg')
        plt.imsave(resultsdir+filenames[j]+'_detout.jpg', image)
        #plt.clf()
        # put detections on image
        nmy, nmx = detection.nonmaxsuppresion(image)
        #print detargs['patchsize']
        patches = detection.getpatches(nmy, nmx, testdata[j], detargs['patchsize'])
        #patches = detection.getpatches(nmy, nmx, det_data.testdata[j], detargs['patchsize'])
        # for each detection in image:
        for i in range(len(nmy)):
            # extract patch around detection
            patch = patches[(nmy[i],nmx[i])]
            #patch = preprocess(np.expand_dims(patch,0))
            patch = np.expand_dims(patch,0)
            # run classification net on patch
            with clsgraph.as_default():
                clssaver.restore(clssess, tf.train.latest_checkpoint(clssessiondir))
                clsres = clssess.run([clsy], feed_dict={clsx: patch})
            classification = np.argmax(clsres[0], 1)[0]
            #print clsres[0]
            ylist, xlist = classifications[classification]
            ylist.append(nmy[i])
            xlist.append(nmx[i])
            classifications[classification] = (ylist,xlist)
            
            # save coordinates, class and test image entry in file
            #string = imagename + '\t' + str(nmy[i]) + '\t' + str(nmx[i]) + '\t' + str(classification)
            
        
        # save image with detections overlaid in different colors
        
        #plt.imshow(original)
        
        colors = ['b', 'g', 'r']
        #ax = plt.Axes(fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #fig.add_axes(ax)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.imshow(det_data.testdata[j], aspect='normal')
        for i in range(classargs['nouts']):
            cly, clx = classifications[i]
            ax.scatter(clx, cly, c=colors[i], s=20, marker='+')
            print len(cly)
        fig.savefig(resultsdir+filenames[j]+'_detclassifications.jpg')
        plt.clf()
        #ax.clear()
        plt.close('all')
    