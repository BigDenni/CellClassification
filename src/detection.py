# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:12:25 2015

@author: dennis
"""

from task import Task
import tensorflow as tf
import scipy.ndimage
import numpy as np

class Detection(Task):
    
    def augment_images(self, traindata, labeldata):
        from elastic_deformation import elastic_transform2
        import random as rd
        import numpy as np
        from scipy import ndimage
    
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
            #newimg = img
            
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
    
    def read_training_sets(self, train_dir, label_dir, _):
    
        import os
        import numpy as np
        import matplotlib.image as mpimg
        
        det_data = self.dataset
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
        
        #det_data.trainlabels = det_data.trainlabels * 100
        #det_data.vallabels = det_data.vallabels * 10
        det_data.trainlabels = det_data.trainlabels
        det_data.vallabels = det_data.vallabels
        
        return det_data
        
    def read_testing_sets(self, test_dir):
    
        import os
        #import numpy as np
        #import matplotlib.image as mpimg
        import skimage.io
        
        det_data = self.dataset
        testdata = []
        counter = 0
        filenames = {}
        
        for filename in os.listdir(test_dir):
            impath = os.path.join(test_dir, filename)
            #img = mpimg.imread(impath)
            img = skimage.io.imread(impath, plugin='tifffile')
            testdata.append(img)
            filenames[counter] = filename
            counter = counter + 1
    
        det_data.testdata = np.array(testdata)
        
        return det_data, filenames
        
    def validate(self, outs, oridata, _, resultsdir, taskargs):
        patchflag = taskargs['patchflag']
        patchsize = taskargs['patchsize']       
        visflag = taskargs['visflag']
        
        if visflag:
            #import skimage.morphology
            import os
            #import numpy as np
            #import scipy.ndimage
            import matplotlib.pyplot as plt
            from matplotlib import cm
            #import skimage.measure
            
            for j in range(len(outs)):
                
                image = outs[j][0,:,:,0]
                image = scipy.ndimage.gaussian_filter(image, sigma=(1, 1), order=0)
                original = oridata[j]
                #print np.max(image)
                #print np.mean(image)
                ##### CC procedure
                '''
                image3 = np.where(image > 20, 1, 0)
                L = skimage.measure.label(image3, background=0)
                L = L+1
                image3 = skimage.morphology.remove_small_objects(L)
                image3 = np.where(image3 > 0, 1, 0)
                image3 = scipy.ndimage.morphology.binary_fill_holes(image3)
                newim = skimage.morphology.binary_erosion(image3)
                L = skimage.measure.label(newim, background=0)
                L = L+1
                newim = skimage.morphology.remove_small_objects(L)
                centers = scipy.ndimage.measurements.center_of_mass(newim, newim, np.unique(newim))
                xp = [tup[1] for tup in centers]
                yp = [tup[0] for tup in centers]
                '''
                
                
                ####### non max suppression procedure
                #maxed = scipy.ndimage.filters.maximum_filter(image, 20)
                #mask = (image == maxed)
                #imagenm = image * mask
                #nmy, nmx = np.where(imagenm > 10)
                nmy, nmx = nonmaxsuppresion(image)
                
                
                ####### extract patches maybe
                if patchflag:
                    patches = getpatches(nmy, nmx, original, patchsize)
                    #padded = np.pad(original, ((patchsize/2,patchsize/2),(patchsize/2,patchsize/2),(0,0)), mode='reflect')
                    if not os.path.exists(resultsdir+'/patches'):
                        os.mkdir(resultsdir+'/patches')
                    for i in range(len(nmy)):
                        #py = nmy[i]
                        #px = nmx[i]
                        #patch = np.copy(padded[py:py+patchsize,px:px+patchsize,:])
                        patch = patches[(nmy[i],nmx[i])]
                        plt.imsave(resultsdir+'/patches/'+str(j)+'_'+str(i)+'.jpg', patch)
            
                fig = plt.figure(frameon=False)
                #plt.imshow(original)
                
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image, aspect='normal')
                ax.scatter(nmx, nmy, c='r', s=10)
                fig.savefig(resultsdir+str(j)+'_detections.jpg')
                #plt.clf()
                ax.imshow(original, aspect='normal')
                ax.scatter(nmx, nmy, c='r', s=10)
                fig.savefig(resultsdir+str(j)+'_original.jpg')
                plt.close('all')
                #plt.imsave(resultsdir+str(index)+'_segmentation.jpg', newim)
                #plt.imsave(resultsdir+str(index)+'_nonmax.jpg', imagenm)
                #plt.imsave(resultsdir+str(index)+'_outfile.jpg', image)
                #plt.imsave(resultsdir+str(index)+'_original.jpg', original)
                
                #print np.max(image)
                #print np.min(image)
                #print 'ey'
                #plt.imsave(resultsdir+str(j)+'_output.jpg', image)
                #plt.imsave(resultsdir+str(j)+'_original.jpg', original)
    
    def loss(self, y_conv):
        y_ = tf.placeholder("float", shape=[None, None, None, 1])
        diff = tf.sub(y_,y_conv)
        loss = tf.reduce_sum(tf.mul(diff,diff))
        return y_, loss
        
def nonmaxsuppresion(image):
    maxed = scipy.ndimage.filters.maximum_filter(image, 20)
    mask = (image == maxed)
    imagenm = image * mask
    return np.where(imagenm > 4)
    
def getpatches(nmy, nmx, original, patchsize):
    dic = {}
    padded = np.pad(original, ((patchsize/2,patchsize/2),(patchsize/2,patchsize/2),(0,0)), mode='reflect')
    for i in range(len(nmy)):
        py = nmy[i]
        px = nmx[i]
        patch = np.copy(padded[py:py+patchsize,px:px+patchsize,:])
        dic[(nmy[i],nmx[i])] = patch
    return dic