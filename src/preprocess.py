# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:56:05 2015

@author: dennis
"""

import numpy as np

def preprocess(data):
    #chmeans = []
    #chstds = []
    
    newdata = np.zeros(data.shape)
    #print np.min(newdata)
    #for i in range(newdata.shape[0]):
    for i in range(newdata.shape[3]):
            #newdata[i,:,:,j] = (newdata[i,:,:,j] - np.mean(newdata[i,:,:,j])) / np.std(newdata[i,:,:,j])
        #chmeans.append(np.mean(newdata[:,:,:,i]))
        #chstds.append(np.std(newdata[:,:,:,i]))
        batch = data[:,:,:,i]
        newdata[:,:,:,i] = np.true_divide(np.subtract(batch, np.mean(batch)),np.std(batch))
        #newdata[:,:,:,i] = (newdata[:,:,:,i] - 0.422317) / 0.175705
        
    #for i in range(newdata.shape[0]):
    #    for j in range(newdata.shape[3]):
    #        image = newdata[i,:,:,j]
    #        newdata[i,:,:,j] = np.divide((image - np.mean(image)), np.std(image))
    return newdata