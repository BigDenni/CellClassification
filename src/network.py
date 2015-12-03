import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def mynet(x, xsize):
    #x = tf.placeholder("float", shape=[None, None, None, 3])
    #y_ = tf.placeholder("float", shape=[None, None, None, 1])
                            
    
    W_conv1 = weight_variable([3, 3, 3, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = conv2d(x, W_conv1) + b_conv1
    h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))
    
    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_pool2 = tf.nn.relu(max_pool_2x2(h_conv2))
    
    #h_resi3 = tf.image.resize_images(h_pool2, tf.shape(x), tf.shape(x))
    h_resi3 = tf.image.resize_bilinear(h_pool2, xsize/2)
    W_conv3 = weight_variable([3, 3, 32, 16])
    b_conv3 = bias_variable([16])
    h_conv3 = tf.nn.relu(conv2d(h_resi3, W_conv3) + b_conv3)
    
    #h_resi4 = tf.image.resize_images(h_conv3, 102, 102)
    h_resi4 = tf.image.resize_bilinear(h_conv3, xsize+2)
    W_conv4 = weight_variable([3, 3, 16, 1])
    b_conv4 = bias_variable([1])
    h_conv4 = conv2d(h_resi4, W_conv4) + b_conv4
    
    return h_conv4
    
def mynet2(x, xsize):

    W_conv1 = weight_variable([3, 3, 3, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = conv2d(x, W_conv1) + b_conv1
    h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))
    
    h_resi2 = tf.image.resize_bilinear(h_pool1, xsize+2)
    W_conv2 = weight_variable([3, 3, 16, 1])
    b_conv2 = bias_variable([1])
    h_conv2 = conv2d(h_resi2, W_conv2) + b_conv2
    
    return h_conv2
    

'''
W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = conv2d(x, W_conv1) + b_conv1
h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))

W_conv2 = weight_variable([3, 3, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_pool2 = tf.nn.relu(max_pool_2x2(h_conv2))

h_resi3 = tf.image.resize_images(h_pool2, traindata.shape[1]/2, traindata.shape[2]/2)
W_conv3 = weight_variable([3, 3, 32, 16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_resi3, W_conv3) + b_conv3)

h_resi4 = tf.image.resize_images(h_conv3, traindata.shape[1]+2, traindata.shape[2]+2)
W_conv4 = weight_variable([3, 3, 16, 1])
b_conv4 = bias_variable([1])
h_conv4 = conv2d(h_resi4, W_conv4) + b_conv4

y_conv = h_conv4
'''
'''
W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = conv2d(x, W_conv1) + b_conv1
h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))

h_resi2 = tf.image.resize_images(h_conv1, traindata.shape[1]+2, traindata.shape[2]+2)
W_conv2 = weight_variable([3, 3, 16, 1])
b_conv2 = bias_variable([1])
h_conv2 = conv2d(h_resi2, W_conv2) + b_conv2

y_conv = h_conv2
'''
'''
W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = conv2d(x, W_conv1) + b_conv1
h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))

W_conv2 = weight_variable([3, 3, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_pool2 = tf.nn.relu(max_pool_2x2(h_conv2))

W_conv3 = weight_variable([3, 3, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3
h_pool3 = tf.nn.relu(max_pool_2x2(h_conv3))

h_resi4 = tf.image.resize_images(h_pool3, det_data.traindata.shape[1]/4, det_data.traindata.shape[2]/4)
W_conv4 = weight_variable([3, 3, 64, 32])
b_conv4 = bias_variable([32])
h_conv4 = conv2d(h_resi4, W_conv4) + b_conv4

h_resi5 = tf.image.resize_images(h_conv4, det_data.traindata.shape[1]/2, det_data.traindata.shape[2]/2)
W_conv5 = weight_variable([3, 3, 32, 16])
b_conv5 = bias_variable([16])
h_conv5 = tf.nn.relu(conv2d(h_resi5, W_conv5) + b_conv5)

h_resi6 = tf.image.resize_images(h_conv5, det_data.traindata.shape[1]+2, det_data.traindata.shape[2]+2)
W_conv6 = weight_variable([3, 3, 16, 1])
b_conv6 = bias_variable([1])
h_conv6 = conv2d(h_resi6, W_conv6) + b_conv6

y_conv = h_conv6
'''
# End of net