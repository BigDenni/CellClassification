# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:15:52 2015

@author: dennis
"""

import numpy
from random import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.


    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = numpy.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)
    
def elastic_transform2(image, gx, gy, s):
    """Elastic deformation as in pulling the image in a random coordinate
    """

    #if random_state is None:
    #    random_state = numpy.random.RandomState(None)
    shape = image.shape
    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    #s = random()*0.003
    dx = (gx - x)
    dy = (gy - y)
    dx = numpy.sign(dx)*dx**2
    dy = numpy.sign(dy)*dy**2
    dx = dx*s
    dy = dy*s
    
    #indices = 
    
    #dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    if len(shape) == 3:
        ch1 = map_coordinates(image[:,:,0], indices, order=1).reshape(image[:,:,0].shape)
        ch2 = map_coordinates(image[:,:,1], indices, order=1).reshape(image[:,:,0].shape)
        ch3 = map_coordinates(image[:,:,2], indices, order=1).reshape(image[:,:,0].shape)
        return numpy.dstack((ch1,ch2,ch3))

    return map_coordinates(image, indices, order=1).reshape(shape)