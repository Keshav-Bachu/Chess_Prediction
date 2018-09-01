# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 08:21:36 2018

@author: Keshav Bachu
"""
#References:
#           https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/


import tensorflow as tf
import numpy as np

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

"""
CNN layer builder, used as a building block for the model

Args:
    input_data: data used to propogate prev layer to next layer
    num_input_chanels: prev layer channels 
    filter_shape: shape of cnn filter
    num_filters: Number of filters used to eval input, translates to the num of channels outputted
    
Returns:
    out_layer: CNN outpt after running the filters
    weights: Weights used as for training
    bias: Biases used for training
"""

def conv_net(input_data, num_input_channels, filter_shape, num_filters):
    #weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    conv_filt_shape = [filter_shape,filter_shape, num_input_channels, num_filters]
    
    weights = create_weights(conv_filt_shape)
    bias = create_biases(num_filters)
    
    out_layer = tf.nn.conv2d(input=input_data, filter= weights, strides= [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.max_pool(value=out_layer, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
    out_layer = tf.nn.relu(out_layer)
    
    
    return out_layer, weights, bias

"""
Partial step in convertint CNN layer to a fully connected layer and thus must by flattened first

Args:
    layer: CNN layer that needs to be flattened

Returns:
    layer: The input layer reshaped to be flat, of size [# examples, sum of all values from prev layer]
"""
def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer

"""
Fully connected layer of the NN used after flattening the CNN

Args:
    input: Input of the previous layer that will be used to build network further
    num_inputs: Num of neuron inputs of the network
    num_outputs: Output neurons of the network
    use_relu: Toggle option for the relu function

Returns:
    layer: Propogated next layer with as many neurons as specified in the num_output
    weights: Weights used in the training of this layer
    biases: Bias values used in this layer
"""
def fc_layer(input,num_inputs,num_outputs, use_relu = False):
    weights = create_weights(shape=[num_inputs, num_outputs])
    
    biases = create_biases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if(use_relu == True):
        layer = tf.nn.relu(layer)
        
    return layer, weights, biases

"""
Used to call the functions above and put the pieces together

Args: 
    xTrain: input data
    yTrain: output answers for data
    learning_rate: variable used for amount of change in gradient decent
    itterations: # of cycles the model trains on
    batch: used for minibatches, which are not implimented at the moment

Returns:
    pred: Numpy array of the predictions from the input data, shape of [# examples, 1]
    weightsTemp: List of all the weights used in the training of the network
    biasTemp: List of all the biases used in the training of the network
    
    Note: The weights and biases were returned so they can be stored for later use
    
"""
def model(xTrain, yTrain, xTest = np.zeros((0)), yTest = np.zeros((0)), learning_rate = 0.01, itterations = 500, batch = 1):
    costs = []
    weights_store = []
    biases_store = []
    
    #set the placeholders for the x and y data
    x = tf.placeholder(tf.float32, shape = [None, 8,8,6], name = 'x')
    y = tf.placeholder(tf.float32, [None, 1], name = 'y')
    
    #make 2 layers of the CNN
    layer1, weightTemp, biasTemp = conv_net(x, 6, 8, 10)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    layer2, weightTemp, biasTemp = conv_net(layer1, 10, 2, 20)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    #flatten for use in fully connected layer
    flattened = flatten(layer2)
    
    #traditional NN fully connected layers
    fully_connected, weightTemp, biasTemp = fc_layer(flattened, flattened.get_shape()[1:4].num_elements(), 16)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    fully_connected2, weightTemp, biasTemp = fc_layer(fully_connected, fully_connected.get_shape()[1:4].num_elements(), 1)
    weights_store.append(weightTemp)
    biases_store.append(biasTemp)
    
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = fully_connected2,labels=y)
    
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    #model training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        temp_cost = 0
        for itter in range (itterations):
            _,temp_cost, check = sess.run([optimizer, cost, fully_connected], feed_dict={x:xTrain, y: yTrain})
            
            if(itter % 100 == 0):
                print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                
            costs.append(temp_cost)
            
        predTF = tf.nn.sigmoid(fully_connected2)
        pred, weightsTemp, biasTemp = sess.run([predTF, weights_store, biases_store], feed_dict={x:xTrain, y: yTrain}) 
        
        if(xTest.shape == (0,) and yTest.shape == (0,)):
            return pred, weightsTemp, biasTemp
        
        else:
            testPrediction = sess.run(predTF, feed_dict={x:xTest, y: yTest})
            return pred, testPrediction, weightsTemp, biasTemp