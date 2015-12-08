# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:12:25 2015

@author: dennis
"""

from task import Task
import tensorflow as tf

class Classification(Task):
    
    def augment_images(self):
        print 'HEJ'
    
    def read_data_sets(self):
        pass
    
    def loss(self, y_conv):
        y_ = tf.placeholder("float", shape=[None, None])
        loss = -tf.reduce_sum(y_*tf.log(y_conv))
        return y_, loss