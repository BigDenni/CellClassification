import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, pad):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=pad)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
def conv(inp, oldfilt, newfilt, filtsize, pad):
    W = weight_variable([filtsize, filtsize, oldfilt, newfilt])
    b = bias_variable([newfilt])
    conv = conv2d(inp, W, pad) + b
    return conv
    
def convrelu(inp, oldfilt, newfilt, filtsize, pad):
    W = weight_variable([filtsize, filtsize, oldfilt, newfilt])
    b = bias_variable([newfilt])
    conv = conv2d(inp, W, pad) + b
    return tf.nn.relu(conv)
    
def convpoolrelu(inp, oldfilt, newfilt, filtsize, pad):
    W = weight_variable([filtsize, filtsize, oldfilt, newfilt])
    b = bias_variable([newfilt])
    conv = conv2d(inp, W, pad) + b
    pool = max_pool_2x2(conv)
    return tf.nn.relu(pool)

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
    
def shorty(x, xsize):

    conv1 = convrelu(x, 3, 16, 3, 'VALID')
    resi = tf.image.resize_bilinear(conv1, xsize+2)
    conv2 = convrelu(resi, 16, 1, 3, 'VALID')
    
    
    #h_resi2 = tf.image.resize_bilinear(h_pool1, xsize+2)
    #W_conv2 = weight_variable([3, 3, 16, 1])
    #b_conv2 = bias_variable([1])
    #h_conv2 = conv2d(h_resi2, W_conv2) + b_conv2
    
    return conv2
    
def deepy(x, xsize):
    conv1 = convrelu(x, 3, 16, 3, 'VALID')
    conv2 = convrelu(conv1, 16, 20, 3, 'VALID')
    conv3 = convrelu(conv2, 20, 24, 3, 'VALID')
    conv4 = convrelu(conv3, 24, 32, 3, 'VALID')
    resi = tf.image.resize_bilinear(conv4, xsize+2)
    conv5 = conv(resi, 32, 1, 3, 'VALID')
    return conv5
    
def wayback(x):
    conv1 = convpoolrelu(x, 3, 16, 3, 'VALID')
    conv2 = convpoolrelu(conv1, 16, 32, 3, 'VALID')
    #up1 = tf.image.resize_bilinear(conv2, xsize/2-4)
    up1 = tf.image.resize_bilinear(conv2, tf.shape(x)[1:3]/2)
    conv3 = convrelu(up1, 32, 16, 3, 'SAME')
    #up2 = tf.image.resize_bilinear(conv3, xsize)
    up2 = tf.image.resize_bilinear(conv3, tf.shape(x)[1:3])
    conv4 = convrelu(up2, 16, 1, 3, 'SAME')
    return conv4

def classy(x, taskargs):
    conv1 = convpoolrelu(x, 3, 16, 3, 'VALID')
    conv2 = convpoolrelu(conv1, 16, 32, 3, 'VALID')
    conv3 = convpoolrelu(conv2, 32, 32, 3, 'VALID')
    conv4 = convpoolrelu(conv3, 32, 48, 3, 'VALID')
    conv5 = convrelu(conv4, 48, taskargs['nouts'], 5, 'VALID')
    resh = tf.reshape(conv5, [-1, taskargs['nouts']])
    softm = tf.nn.softmax(resh)
    return softm
    
def network(x, netname, taskargs):
    '''
    dic = {
        'shorty': shorty(x, xsize),
        'deepy': deepy(x, xsize),
        'wayback': wayback(x, xsize),
        'classy': classy(x, taskargs)
    }
    if netname in dic:
        return dic[netname]
    else:
        print 'Not valid netname'   
    '''
    if netname == 'wayback':
        return wayback(x)
    elif netname == 'classy':
        return classy(x, taskargs)
    else:
        print 'Not valid netname'
    
# End of net
