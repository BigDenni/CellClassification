# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:03:24 2015

@author: dennis
"""

class dataset:
    pass

def read_data_sets(train_dir, label_dir):
    
    import os
    import numpy as np
    import matplotlib.image as mpimg
    
    det_data = dataset()
    traindata = []
    valdata = []
    trainlabel = []
    vallabel = []
    counter = 1
    
    for filename in os.listdir(train_dir):
        impath = os.path.join(train_dir, filename)
        labelim = 'labels_' + str.split(filename, '_')[1];
        labelpath = os.path.join(label_dir, labelim);
        img = mpimg.imread(impath)
        labelimg = mpimg.imread(labelpath)
        if counter % 2 == 0:
            #if counter < 10:
            #    print filename
            valdata.append(img)
            vallabel.append(labelimg)
        else:
            traindata.append(img)
            trainlabel.append(labelimg)
        counter = counter+1
        
    det_data.traindata = np.array(traindata)
    det_data.valdata = np.array(valdata)
    det_data.trainlabels = np.array(trainlabel)
    det_data.trainlabels = np.reshape(det_data.trainlabels, [det_data.trainlabels.shape[0],det_data.trainlabels.shape[1],det_data.trainlabels.shape[2],1])
    det_data.vallabels = np.array(vallabel)
    det_data.vallabels = np.reshape(det_data.vallabels, [det_data.vallabels.shape[0],det_data.vallabels.shape[1],det_data.vallabels.shape[2],1])
    
    return det_data
    
def read_data_sets_testing(test_dir):
    
    import os
    import numpy as np
    #import matplotlib.image as mpimg
    import skimage.io
    
    det_data = dataset()
    testdata = []
    
    for filename in os.listdir(test_dir):
        impath = os.path.join(test_dir, filename)
        #img = mpimg.imread(impath)
        img = skimage.io.imread(impath, plugin='tifffile')
        testdata.append(img)

    det_data.testdata = np.array(testdata)
    
    return det_data