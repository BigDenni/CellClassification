# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:56:05 2015

@author: dennis
"""

import numpy as np

def preprocess(data):
    #chmeans = []
    #chstds = []
    
    newdata = np.copy(data)
    for i in range(newdata.shape[3]):
        #chmeans.append(np.mean(newdata[:,:,:,i]))
        #chstds.append(np.std(newdata[:,:,:,i]))
        newdata[:,:,:,i] = (newdata[:,:,:,i] - np.mean(newdata[:,:,:,i])) / np.std(newdata[:,:,:,i])
        
    #for i in range(newdata.shape[0]):
    #    for j in range(newdata.shape[3]):
    #        image = newdata[i,:,:,j]
    #        newdata[i,:,:,j] = np.divide((image - np.mean(image)), np.std(image))
    return newdata