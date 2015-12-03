# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:27:30 2015

@author: dennis
"""

from elastic_deformation import elastic_transform2
import random as rd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import ndimage

def augment_data_sets2(train_dir, label_dir, augment_folder):
    
    import os
    
    #os.mkdir(augment_folder)

    
    for filename in os.listdir(train_dir):
        impath = os.path.join(train_dir, filename)
        labelim = 'labels_' + str.split(filename, '_')[1];
        labelpath = os.path.join(label_dir, labelim);
        img = mpimg.imread(impath)
        #img = img[:,:,0]
        labelimg = mpimg.imread(labelpath)
        for i in range(1):
            newimg = img
            
            s = rd.random()*0.003
            gx = rd.random()*img.shape[1]
            gy = rd.random()*img.shape[0]
            newimg = elastic_transform2(img,gx,gy,s)
            labelimg = elastic_transform2(labelimg,gx,gy,s)
            
            angle = rd.randint(0,3)*90
            newimg = ndimage.rotate(newimg,angle)
            labelimg = ndimage.rotate(labelimg,angle)
            if rd.random() > 0.5:
                newimg = np.flipud(newimg)
                labelimg = np.flipud(labelimg)
            if rd.random() > 0.5:
                newimg = np.fliplr(newimg)
                labelimg = np.fliplr(labelimg)
            
            plt.imsave(augment_folder+'/'+filename+str(i)+'.jpg', newimg)
            plt.imsave(augment_folder+'/'+filename+str(i)+'_label.jpg', labelimg, cmap = cm.gray)
        plt.imsave(augment_folder+'/'+filename+'_original.jpg', img)
        
def augment_data_sets(traindata, labeldata):
    
    
    #os.mkdir(augment_folder)
    augtrain = np.zeros(traindata.shape)
    auglabel = np.zeros(labeldata.shape)
    
    for i in range(traindata.shape[0]):
        
        img = np.copy(traindata[i,:,:,:])
        #img = img[:,:,0]
        labelimg = np.copy(labeldata[i,:,:,0])
        
    
        s = rd.random()*0.003
        gx = rd.random()*img.shape[1]
        gy = rd.random()*img.shape[0]
        newimg = elastic_transform2(img,gx,gy,s)
        labelimg = elastic_transform2(labelimg,gx,gy,s)
        
        angle = rd.randint(0,3)*90
        newimg = ndimage.rotate(newimg,angle)
        labelimg = ndimage.rotate(labelimg,angle)
        if rd.random() > 0.5:
            newimg = np.flipud(newimg)
            labelimg = np.flipud(labelimg)
        if rd.random() > 0.5:
            newimg = np.fliplr(newimg)
            labelimg = np.fliplr(labelimg)
        
        augtrain[i,:,:,:] = newimg
        auglabel[i,:,:,:] = labelimg[:,:,np.newaxis]
        
    return augtrain, auglabel
    
train_dir = '/home/dennis/CellClassification/data/working_folder/training'
label_dir = '/home/dennis/CellClassification/data/working_folder/labels'
augment_folder = '/home/dennis/CellClassification/data/working_folder/augment'
#train_dir = 'F:\\LinuxMintDocuments\\Cell_Detection_Classification\\data\\working_folder\\training'
#label_dir = 'F:\\LinuxMintDocuments\\Cell_Detection_Classification\\data\\working_folder\\labels'
#augment_folder = 'F:\\LinuxMintDocuments\\Cell_Detection_Classification\\data\\working_folder\\augment'

augment_data_sets2(train_dir, label_dir, augment_folder)