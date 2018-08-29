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
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


with open('white_win.txt', 'r') as file:
    white_wins = file.read()
    
    
white_wins = white_wins.split('= = = = = = = =\n')
for i in range(len(white_wins)):
    white_wins[i] = white_wins[i].split("- - - - - - - -\n")

gameList = []
turnList = []
for game in range(len(white_wins)):
    turnList = []
    for turn in range(len(white_wins[game])):
        white_wins[game][turn] = list(white_wins[game][turn])
        if ' ' in white_wins[game][turn]:
            #white_wins[game][turn].remove(' ')
            white_wins[game][turn][:] = [x for x in white_wins[game][turn] if x != ' ']
        for char in range(len(white_wins[game][turn])):
            white_wins[game][turn][char] = ord(white_wins[game][turn][char])
        #white_wins[game][turn] = white_wins[game][turn].remove(32)
        white_wins[game][turn][:] = [x for x in white_wins[game][turn] if x != 10]
        
        if(len(white_wins[game][turn]) != 0):
            temp = np.asarray(white_wins[game][turn], np.float32)
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

whiteWinFormatted = []
x = 0
for game in range(len(gameList)):
    whiteWinFormatted.append(gameList[game][0])
    for turn in range (1, 6):
        whiteWinFormatted[x] = np.dstack((whiteWinFormatted[x], gameList[game][turn]))
    x = x + 1

#whiteWinFormatted is 8x8x6 , 6 representing the last 6 turns, now the data can be sent to a conv NN
whiteWinFormatted = np.asarray(whiteWinFormatted)
wWin = np.zeros((19,1))
wWin = wWin + 1





with open('black_win.txt', 'r') as file:
    white_wins = file.read()    
white_wins = white_wins.split('= = = = = = = =\n')
for i in range(len(white_wins)):
    white_wins[i] = white_wins[i].split("- - - - - - - -\n")

gameList = []
turnList = []
for game in range(len(white_wins)):
    turnList = []
    for turn in range(len(white_wins[game])):
        white_wins[game][turn] = list(white_wins[game][turn])
        if ' ' in white_wins[game][turn]:
            #white_wins[game][turn].remove(' ')
            white_wins[game][turn][:] = [x for x in white_wins[game][turn] if x != ' ']
        for char in range(len(white_wins[game][turn])):
            white_wins[game][turn][char] = ord(white_wins[game][turn][char])
        #white_wins[game][turn] = white_wins[game][turn].remove(32)
        white_wins[game][turn][:] = [x for x in white_wins[game][turn] if x != 10]
        
        if(len(white_wins[game][turn]) != 0):
            temp = np.asarray(white_wins[game][turn], np.float32)
            #print(temp.shape, ' ', game)
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

blackWinFormatted = []
x = 0
for game in range(len(gameList)):
    blackWinFormatted.append(gameList[game][0])
    for turn in range (1, 6):
        blackWinFormatted[x] = np.dstack((blackWinFormatted[x], gameList[game][turn]))
    x = x + 1

#blackWinFormatted is 8x8x6 , 6 representing the last 6 turns, now the data can be sent to a conv NN
blackWinFormatted = np.asarray(blackWinFormatted)
bWin = np.zeros((19,1))
bWin = bWin + 0

allX = np.append(blackWinFormatted, whiteWinFormatted, 0)
allY = np.append(bWin, yWin, 0)
allX, allY = shuffle_in_unison(allX, allY)

prediction = CCT.model(allX,allY)
prediction = prediction[0]
prediction = prediction.reshape[prediction.shape[0],]
prediction = np.ndarray.tolist(prediction[0])



