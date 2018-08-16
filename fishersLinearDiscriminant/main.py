#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
import numpy as np
import loadData
import generateData
import fisher


def binaryFisher():
    filePath = "iris.data.txt"
    bX, bY = loadData.DataLoader(filePath).binaryClassifyGetData()
    bX = np.array(bX)
    bY = np.array(bY).reshape(100, 1)  # 用reshape函数修改Y矩阵的形式
    trainX, trainY, testX, testY = generateData.DataGenerator(bX, bY).randDivide()
    trainX = trainX.astype(float)
    trainY = trainY.astype(float)
    testX = testX.astype(float)
    testY = testY.astype(float)
    # print(trainX)
    # print(trainY)
    # print(testX)
    # print(testY)
    # print(np.shape(trainX), np.shape(trainY), np.shape(testX), np.shape(testY))
    miu1, length1 = fisher.FisherLinearDiscriminant(trainX, trainY, testX, testY).getMiu(1)
    miu2, length2 = fisher.FisherLinearDiscriminant(trainX, trainY, testX, testY).getMiu(2)
    # # print(miu1)
    # # print(miu2)
    # S1 = np.zeros((4, 4))
    # S2 = np.zeros((4, 4))
    # for i in range(len(trainX)):
    #     if int(trainY[i]) == 1:
    #         xMat = (trainX[i] - miu1).reshape((4, 1))
    #         s = np.dot(xMat, xMat.T)
    #         # print(s)
    #         S1 = S1 + s
    #     else:
    #         xMat = (trainX[i] - miu2).reshape((4, 1))
    #         s = np.dot(xMat, xMat.T)
    #         # print(s)
    #         S2 = S2 + s
    # # print(S1)
    # # print(S2)
    S1 = fisher.FisherLinearDiscriminant(trainX, trainY, testX, testY).getSw(miu1, 1)
    S2 = fisher.FisherLinearDiscriminant(trainX, trainY, testX, testY).getSw(miu2, 2)
    Sw = np.mat(S1 + S2)
    miu = (miu2 - miu1).reshape(4, 1)
    w = np.dot(Sw.I, miu)
    # print(w)
    fisher.FisherLinearDiscriminant(trainX, trainY, testX, testY).plot(w)
    miuAll = (length1 * miu1 + length2 * miu2)/(length1 + length2)
    fisher.FisherLinearDiscriminant(trainX, trainY, testX, testY).modelTest(w, miuAll)


def multiFisher():
    filePath = "iris.data.txt"
    bX, bY = loadData.DataLoader(filePath).multiClassifyGetData()
    mX = np.array(bX)
    mY = np.array(bY).reshape(150, 1)  # 用reshape函数修改Y矩阵的形式
    trainX, trainY, testX, testY = generateData.DataGenerator(mX, mY).randDivide()
    # print(trainX)
    # print(trainY)
    # print(testX)
    # print(testY)
    # print(np.shape(trainX), np.shape(trainY), np.shape(testX), np.shape(testY))


if __name__ == "__main__":
    print("1. binary Fisher")
    print("2. multi-class Fisher")
    if (input("Choose which function to do:")) == "1":
        binaryFisher()
    else:
        multiFisher()

