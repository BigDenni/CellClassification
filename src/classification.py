# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:12:25 2015

@author: dennis
"""

from task import Task
import tensorflow as tf

class Classification(Task):
    
    def augment_images(self, traindata, labeldata):
        from elastic_deformation import elastic_transform2
        import random as rd
        import numpy as np
        from scipy import ndimage
    
        #os.mkdir(augment_folder)
        augtrain = np.zeros(traindata.shape)
        
        for i in range(traindata.shape[0]):
            
            img = np.copy(traindata[i,:,:,:])
        
            s = rd.random()*0.003
            gx = rd.random()*img.shape[1]
            gy = rd.random()*img.shape[0]
            newimg = elastic_transform2(img,gx,gy,s)
            
            angle = rd.randint(0,3)*90
            newimg = ndimage.rotate(newimg,angle)
            if rd.random() > 0.5:
                newimg = np.flipud(newimg)
            if rd.random() > 0.5:
                newimg = np.fliplr(newimg)
            
            augtrain[i,:,:,:] = newimg
            
        return augtrain, labeldata
    
    def read_training_sets(self, train_dir, _, taskargs):
        import os
        import numpy as np
        import matplotlib.image as mpimg
        
        det_data = self.dataset
        traindata = []
        valdata = []
        trainlabel = []
        vallabel = []
        counter = 1
        
        labels = 'annotations.txt'
        labelpath = os.path.join(train_dir, labels);
        labeldic = {}
        with open(labelpath) as f:
            for line in f:
                s = line.split()
                labeldic[s[0]] = s[1]
        
        labelmaps = taskargs['labelmaps']
        for filename in os.listdir(train_dir):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".tif"):
                impath = os.path.join(train_dir, filename)
                img = mpimg.imread(impath)
                label = np.zeros(taskargs['nouts'])
                label[labelmaps[int(labeldic[filename])]] = 1

                if counter % 2 == 0:
                    #if counter < 10:
                    #    print filename
                    valdata.append(img)
                    vallabel.append(label)
                else:
                    traindata.append(img)
                    trainlabel.append(label)
                counter = counter+1
            
        det_data.traindata = np.array(traindata)
        det_data.valdata = np.array(valdata)
        det_data.trainlabels = np.array(trainlabel)
        det_data.vallabels = np.array(vallabel)
        
        return det_data
    
    def read_testing_sets(self, test_dir):
        import os
        import numpy as np
        #import matplotlib.image as mpimg
        import skimage.io
        
        det_data = self.dataset
        testdata = []
        
        for filename in os.listdir(test_dir):
            impath = os.path.join(test_dir, filename)
            #img = mpimg.imread(impath)
            img = skimage.io.imread(impath, plugin='tifffile')
            testdata.append(img)
    
        det_data.testdata = np.array(testdata)
        
        return det_data
    
    def validate(self, outs, _, labels, resultsdir, taskargs):
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        labelsi = np.argmax(labels, 1)
        #print 'check', labelsi.shape
        outsi = np.argmax(np.array(outs).reshape(len(outs),taskargs['nouts']), 1)
        #print outsi.shape
        #print labels
        conf = confusion_matrix(labelsi, outsi) 
        print conf
        print 'total', np.trace(conf)/float(np.sum(conf))
        for i in range(taskargs['nouts']):
            print 'class ' + str(i) + ' sens', conf[i,i]/float(np.sum(conf[i,:])), 'class ' + str(i) + ' ppv', conf[i,i]/float(np.sum(conf[:,i]))
    
    def loss(self, y_conv):
        y_ = tf.placeholder("float", shape=[None, None])
        loss = -tf.reduce_sum(y_*tf.log(y_conv))
        return y_, loss