#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from random import *


class AicLogistic(object):
    def __init__(self, dataX, dataY, testX, testY):
        self.dataX = dataX
        self.dataY = dataY
        self.testX = testX
        self.testY = testY

    def phi(self):
        p_j = list([])
        p = list([])
        x = self.dataX.shape[0]
        y = self.dataX.shape[1]
        # print(x, y)
        a = 0.0
        for n in range(x):
            for i in range(y):
                for j in range(6):
                    a = pow(float(self.dataX[n, i]), j)
                    p_j.append(a)
            # print(np.shape(p_j))
            p.append(p_j)
            p_j = list([])
        p = np.array(p)
        # print(p)
        # print(np.shape(p))
        return p

    @staticmethod
    def phiX(inX):
        p_j = list([])
        p = list([])
        x = inX.shape[0]
        y = inX.shape[1]
        # print(x, y)
        a = 0.0
        for n in range(x):
            for i in range(y):
                for j in range(6):
                    a = pow(float(inX[n, i]), j)
                    p_j.append(a)
            # print(np.shape(p_j))
            p.append(p_j)
            p_j = list([])
        p = np.array(p)
        # print(p)
        # print(np.shape(p))
        return p

    @staticmethod
    def sigmoid(inX):
        if inX >= 0:
            return 1.0 / (1 + np.exp(-inX))
        else:
            return np.exp(inX) / (1 + np.exp(inX))

    def logisticTrain(self, p, numIter=150):
        m, n = np.shape(self.dataX)
        weights = np.ones(n*6)
        for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4/(1.0+j+i)+0.01
                randIndex = int(uniform(0, len(dataIndex)))
                sumA = sum(p[randIndex]*weights)
                h = self.sigmoid(sumA)
                error = int(self.dataY[randIndex])-2 - h
                weights = weights + alpha * error * p[randIndex]
                del(dataIndex[randIndex])
        return weights

    def logisticTest(self, weights):
        error = 0
        testX = self.phiX(self.testX)
        weights = np.array(weights, float)
        label = []
        # print(testX)
        # print(weights)
        for i in range(len(self.testX)):
            h = self.sigmoid(np.dot(testX[i], weights.T))
            if h > 0.5:
                label.append(3)
            else:
                label.append(2)
            if label[i] != int(self.testY[i]):
                error += 1
        accuracy = (len(self.testX) - error) / float(len(self.testX))
        print("The accuracy of this model is %.2f%%" % (accuracy * 100))
