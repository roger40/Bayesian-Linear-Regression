#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
import numpy as np
import random


class DataGenerator(object):
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []

    def randDivide(self, testRatio=0.3):
        size = len(self.dataY)
        allNum = list(range(size))
        testNum = random.sample(allNum, size)
        for i in range(int(testRatio * size)):
            self.testX.append(self.dataX[testNum[i]])
            self.testY.append(self.dataY[testNum[i]])
        for i in range(int((1-testRatio) * size)):
            self.trainX.append(self.dataX[testNum[-i]])
            self.trainY.append(self.dataY[testNum[-i]])
        return np.array(self.trainX), np.array(self.trainY), np.array(self.testX), np.array(self.testY)
