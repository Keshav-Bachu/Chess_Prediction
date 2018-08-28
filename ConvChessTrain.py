# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 08:21:36 2018

@author: Keshav Bachu
"""

import tensorflow as tf
import numpy as np

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def conv_net(input_data, num_input_channels, filter_shape, num_filters):
    #weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    conv_filt_shape = [8,8,6, 10]
    
    weights = create_weights(conv_filt_shape)
    bias = create_biases(10)
    
    out_layer = tf.nn.conv2d(input=input_data, filter= weights, strides= [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.max_pool(value=out_layer, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
    out_layer = tf.nn.relu(out_layer)
    
    
    return out_layer

def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer

def fc_layer(input,num_inputs,num_outputs):
    weights = create_weights(shape=[num_inputs, num_outputs])
    
    biases = create_biases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.relu(layer)
 
    return layer

def model(xTrain, yTrain, learning_rate = 0.01, itterations = 1000, batch = 1):
    #ops.reset_default_graph()
    costs = []
    x = tf.placeholder(tf.float32, shape = [None, 8,8,6])
    y = tf.placeholder(tf.float32, [None, 1])
    
    layer1 = conv_net(x, 6, 10, 6)
    #layer2 = conv_net(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
    flattened = flatten(layer1)
    
    fully_connected = fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 1)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected,labels=y)
    
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        temp_cost = 0
        for itter in range (itterations):
            _,temp_cost = sess.run([optimizer, cost], feed_dict={x:xTrain, y: yTrain})
            
            if(itter % 100 == 0):
                print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                
            costs.append(temp_cost)
        
    
    
    
    
        

#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py