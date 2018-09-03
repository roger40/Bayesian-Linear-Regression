#!/usr/bin/env python
# -*- coding: utf-8 -*-
import loadData
import dataGenerate
import AIC_Logistic
import showPlt

if __name__ == '__main__':
    lD = loadData.DataLoader("iris.data.txt")
    dataX, dataY = lD.loadData()
    dG = dataGenerate.DataGenerator(dataX, dataY)
    trainX, trainY, testX, testY = dG.randDivide(testRatio=0.3)
    #  print(trainX, "\n", trainY, "\n", testX, "\n", testY)
    aicL = AIC_Logistic.AicLogistic(trainX, trainY, testX, testY)
    p = aicL.phi()
    weights = aicL.logisticTrain(p, numIter=150)
    sP = showPlt.ShowPlt(trainX, trainY, weights)
    sP.plotMap()
    aicL.logisticTest(weights)
