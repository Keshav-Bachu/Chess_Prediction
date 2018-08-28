# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 08:21:36 2018

@author: Keshav Bachu
"""

import tensorflow as tf
import numpy as np
def conv_net(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    #weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    conv_filt_shape = [8,8,6, 10]
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name + 'W')
    bias = tf.Variable(tf.truncated_normal([10]), name= name + 'b')
    
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    
    return out_layer

def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer

def fc_layer(input,num_inputs,num_outputs):
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
    
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
 
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.relu(layer)
 
    return layer

def model(xTrain, yTrain, learning_rate = 0.01, itterations = 1000, batch = 1):
    #ops.reset_default_graph()
    x = tf.placeholder(tf.float32, shape = [None, 8,8,6])
    y = tf.placeholder(tf.float32, [None, 1])
    
    layer1 = conv_net(xTrain, 1, 32, [5, 5], [2, 2], name='layer1')
    #layer2 = conv_net(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
    flattened = flatten(layer1)
    
    fully_connected = fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 1)
    
    
    
    
    
        

#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py