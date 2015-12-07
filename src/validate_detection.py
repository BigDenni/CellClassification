# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:37:22 2015

@author: dennis
"""

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def validate_detection(image, original, index, resultsdir, patchflag=False, patchsize=100):
    #import skimage.morphology
    import os
    #import skimage.measure
    
    
    image = image[0,:,:,0]
    image = scipy.ndimage.gaussian_filter(image, sigma=(5, 5), order=0)
    
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
    maxed = scipy.ndimage.filters.maximum_filter(image, 20)
    mask = (image == maxed)
    imagenm = image * mask
    nmy, nmx = np.where(imagenm > 10)
    
    ####### extract patches maybe
    if patchflag:
        padded = np.pad(original, ((patchsize/2,patchsize/2),(patchsize/2,patchsize/2),(0,0)), mode='reflect')
        if not os.path.exists(resultsdir+'/patches'):
            os.mkdir(resultsdir+'/patches')
        for i in range(len(nmy)):
            py = nmy[i]
            px = nmx[i]
            patch = np.copy(padded[py:py+patchsize,px:px+patchsize,:])
            plt.imsave(resultsdir+'/patches/'+str(index)+'_'+str(i)+'.jpg', patch)

    fig = plt.figure(frameon=False)
    #plt.imshow(original)
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='normal')
    ax.scatter(nmx, nmy, c='r', s=10)
    fig.savefig(resultsdir+str(index)+'_detections.jpg')
    #plt.clf()
    ax.imshow(original, aspect='normal')
    ax.scatter(nmx, nmy, c='r', s=10)
    fig.savefig(resultsdir+str(index)+'_original.jpg')
    plt.close('all')
    #plt.imsave(resultsdir+str(index)+'_segmentation.jpg', newim)
    #plt.imsave(resultsdir+str(index)+'_nonmax.jpg', imagenm)
    #plt.imsave(resultsdir+str(index)+'_outfile.jpg', image)
    #plt.imsave(resultsdir+str(index)+'_original.jpg', original)
    
### TODO
def validate_classification():
    return 0