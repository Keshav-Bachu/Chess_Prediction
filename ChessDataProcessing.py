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
yWin = np.zeros((19,1))
yWin = yWin + 1

CCT.model(whiteWinFormatted,yWin)
