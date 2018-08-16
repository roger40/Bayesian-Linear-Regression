#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class FisherLinearDiscriminant(object):
    def __init__(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

    def getMiu(self, category):
        length = 0
        cateArr = []
        for i in range(len(self.trainX)):
            if int(self.trainY[i]) == category:
                cateArr.append(self.trainX[i])
                length += 1
        cateArr = np.array(cateArr)
        cateArr = cateArr.astype(float)
        Miu = np.mean(cateArr, axis=0)
        return Miu, length

    def getSw(self, miu, category):
        S = np.zeros((4, 4))
        for i in range(len(self.trainX)):
            if int(self.trainY[i]) == category:
                xMat = (self.trainX[i] - miu).reshape((4, 1))
                s = np.dot(xMat, xMat.T)
                # print(s)
                S = S + s
        return S

    def plot(self, w):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        yMat = np.dot(self.trainX, w)
        y = yMat.reshape(1, -1)
        y = y.tolist()
        # print(y)
        for i in range(len(self.trainX)):
            if int(self.trainY[i]) == 1:
                ax.scatter(y[0][i], 0, c='red')
            else:
                ax.scatter(y[0][i], 0, c='blue')
        plt.xlabel("y=w.T*X")
        plt.ylabel("category")
        plt.show()

    def modelTest(self, w, miuAll):
        error = 0
        label = []
        for i in range(len(self.testX)):
            y = np.dot(w.T, (self.testX[i] - miuAll))
            if y > 0:
                label.append(2)
            else:
                label.append(1)
            if label[i] != int(self.testY[i]):
                error += 1
        accuracy = (len(self.testX) - error)/float(len(self.testX))
        print("The accuracy of this model is %.2f%%" % (accuracy*100))

