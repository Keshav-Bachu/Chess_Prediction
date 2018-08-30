#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 08:24:47 2018

@author: keshavbachu
"""

import chess.pgn
import sys
import pandas as pd
import numpy as np
import ConvChessTrain as CCT

#borrowed from
#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
#allows the shufflinf of both data and solution set
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

"""
Format the data for the input of the CNN model by reading data from a text file, specific to chess matches

Args:
    name: Name of the input text file used to read the data
    answer_value: The value output the answer is stored with

Returns:
    formattedData: Numpy array output of the data, with the shape of [# examples, data rows, data cols, 6]
    yOuyput: Numpy array corresponding to the answers of the input data, with the shape of [# examples, 1] 
"""
def load_data(name, answer_value):
    with open(name, 'r') as file:
        _wins = file.read()
        
    
    #seperate by games and turns    
    _wins = _wins.split('= = = = = = = =\n')
    for i in range(len(_wins)):
        _wins[i] = _wins[i].split("- - - - - - - -\n")
    
    #all the games are pushed into a list and formatted to be games with channels representing turns
    #The games that are shorter than 6 turns (the channel that is used here) then the last turn is repeated
    gameList = []
    turnList = []
    for game in range(len(_wins)):
        turnList = []
        for turn in range(len(_wins[game])):
            _wins[game][turn] = list(_wins[game][turn])
            if ' ' in _wins[game][turn]:
                #_wins[game][turn].remove(' ')
                _wins[game][turn][:] = [x for x in _wins[game][turn] if x != ' ']
            for char in range(len(_wins[game][turn])):
                _wins[game][turn][char] = ord(_wins[game][turn][char])
            #_wins[game][turn] = _wins[game][turn].remove(32)
            _wins[game][turn][:] = [x for x in _wins[game][turn] if x != 10]
            
            if(len(_wins[game][turn]) != 0):
                temp = np.asarray(_wins[game][turn], np.float32)
                temp = temp.reshape(8,8)
                turnList.append(temp)
        gameList.append(turnList)
        
    
    for game in range(len(gameList)):
        if(len(gameList[game]) >= 6):
            gameList[game] = gameList[0][len(gameList[0]) - 6:]
        elif(len(gameList[game]) != 0):
            while(len(gameList[game]) < 6):
                gameList[game].append(gameList[game][0])
    
    gameList.remove(gameList[len(gameList) - 1]) 
    
    formattedData = []
    x = 0
    for game in range(len(gameList)):
        formattedData.append(gameList[game][0])
        for turn in range (1, 6):
            formattedData[x] = np.dstack((formattedData[x], gameList[game][turn]))
        x = x + 1
    
    #formattedData is 8x8x6 , 6 representing the last 6 turns, now the data can be sent to a conv NN
    formattedData = np.asarray(formattedData)
    yOutput = np.zeros((formattedData.shape[0],1))
    yOutput = yOutput + answer_value
    return formattedData, yOutput

"""
Calculate the accuracy of the returned predictions

Args:
    allY: The answer key of the values inputted
    predictions: the output from the network, that is compared here
    
Returns:
    accuracy = Accuracy value that is calculated

"""
def compute_Accuracy(allY, predictions):
    totalExample = allY.shape[0]
    totalErrors = np.sum(np.abs(allY - prediction))
    accuracy  = 1 - totalErrors/totalExample

    print("Accuracy of model is: ", accuracy)
    return accuracy



#load and format the white/black win data
whiteWinFormatted, wWin = load_data('white_win.txt', 0)
blackWinFormatted, bWin = load_data('black_win.txt', 1)

#combine the data
allX = np.append(blackWinFormatted, whiteWinFormatted, 0)
allY = np.append(bWin, wWin, 0)
allX, allY = shuffle_in_unison(allX, allY)

#run the NN model, prediction of CONV NN is the output
prediction, weights, biases = CCT.model(allX,allY)
prediction = prediction.astype(int)

compute_Accuracy(allY, prediction)

#np.save("WeightsTrained.npy", weights)
#np.save("BiasesTrained.npy", biases)

#prediction accuracy, push into a funcion or into the model training portion for easier readability





