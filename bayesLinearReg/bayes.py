#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random


def dataLoad():
    file = open('data.txt', 'r+', encoding='utf-8') # 打开文件
    xMat = []
    yMat = []
    for line in file.readlines():  # 逐行读取
        lineArr = line.encode('utf-8').decode('utf-8-sig').strip().split('\t')  # 以制表符来进行分割存表
        xMat.append(float(lineArr[0]))  # 存储数据
        yMat.append(float(lineArr[1]))  # 存储数据
    return xMat, yMat  # 返回输入集合和输出集合


def dataDivide(xMat, yMat):
    """
    这个函数通过将200个下标打乱重新存表的方法来实现对数据集的随机选取
    前60个数据作为测试集
    后140个数据作为训练集
    :param xMat:
    :param yMat:
    :return trainXMat, trainYMat, testXMat, testYMat:
    """
    testXMat = []
    testYMat = []
    trainXMat = []
    trainYMat = []
    allNum = list(range(len(xMat)))
    testNum = random.sample(allNum, 200)
    for i in range(60):
        testXMat.append(xMat[testNum[i]])
        testYMat.append(yMat[testNum[i]])
    for i in range(140):
        trainXMat.append(xMat[testNum[-i]])
        trainYMat.append(yMat[testNum[-i]])
    return trainXMat, trainYMat, testXMat, testYMat


def phi(x):
    """
    将输入x转换成基函数，此处的基函数使用的是幂函数，最高次幂设定为7次
    将幂基函数存储为列表格式返回
    :param x:
    :return p:
    """
    p = list([])
    for i in range(8):
        p.append(pow(x, i))
    return p


def bayesCompare(xMat, yMat):
    """
    首先使用正规方程的方法进行估计，
    再使用贝叶斯线性回归法进行估计，
    再使用从数据中学习参数alpha和beta的贝叶斯线性回归进行估计
    将三种方法进行比较
    :param xMat:
    :param yMat:
    :return:
    """
    PHI = []
    for x in xMat:
        PHI.append(phi(x))
    PHI = np.array(PHI)
    print("PHI = ", PHI)

    # Normal Linear Regression
    omega_normal = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, yMat))
    print("omega(normal linear Regression) = ", omega_normal)

    # Bayesian Linear Regression
    alpha = 0.1  # 设置初始估计值
    beta = 9.0  # 设置初始估计值
    Sigma_N = np.linalg.inv(alpha*np.identity(PHI.shape[1])+beta*np.dot(PHI.T, PHI))  # PRML 公式3.54
    mu_N = beta*np.dot(Sigma_N, np.dot(PHI.T, yMat))  # PRML 公式3.53
    print("mu_N(Bayesian linear regression) = ", mu_N)

    # Bayesian Linear Regression with alpha and beta calculated from data
    alphaT = alpha
    betaT = beta
    while True:
        lame, z = np.linalg.eig(beta*np.dot(PHI.T, PHI))
        gama = 0.0
        for lam in lame:
            gama += float(lam/(lam+alpha))
        Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
        mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, yMat))
        alpha = gama/np.dot(mu_N, mu_N.T)
        nSum = 0.0
        for n in range(len(PHI[0])):
            nSum += (yMat[n] - np.dot(mu_N, PHI[n]))**2
        beta_inv = (1/len(PHI[0]))*nSum
        beta = 1/beta_inv
        if (abs(alpha - alphaT) < 0.001) and (abs(beta - betaT) < 0.001):
            break
        alphaT = alpha
        betaT = beta
    Sigma_NN = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))  # PRML 公式3.54
    mu_NN = beta * np.dot(Sigma_NN, np.dot(PHI.T, yMat))
    print("alpha = %f, beta = %f " % (alpha, beta))

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.scatter(xMat, yMat, s=30, c='red', marker='s')
    x = np.arange(0, 13.0, 0.1)
    y = np.dot(omega_normal, phi(x))
    ax.plot(x, y, c='black')
    bx = fig.add_subplot(132)
    bx.scatter(xMat, yMat, s=30, c='red', marker='s')
    x = np.arange(0, 13.0, 0.1)
    y = np.dot(mu_N, phi(x))
    bx.plot(x, y)
    cx = fig.add_subplot(133)
    cx.scatter(xMat, yMat, s=30, c='red', marker='s')
    x = np.arange(0, 13.0, 0.1)
    y = np.dot(mu_NN, phi(x))
    cx.plot(x, y, c='yellow')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def bayes(xMat, yMat):
    PHI = []
    for x in xMat:
        PHI.append(phi(x))
    PHI = np.array(PHI)
    print("PHI = ", PHI)

    alpha = 0.1  # 设置初始估计值
    beta = 9.0  # 设置初始估计值

    alphaT = alpha
    betaT = beta
    while True:
        lame, z = np.linalg.eig(beta * np.dot(PHI.T, PHI))
        gama = 0.0
        for lam in lame:
            gama += float(lam / (lam + alpha))
        Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
        mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, yMat))
        alpha = gama / np.dot(mu_N, mu_N.T)
        nSum = 0.0
        for n in range(len(PHI[0])):
            nSum += (yMat[n] - np.dot(mu_N, PHI[n])) ** 2
        beta_inv = (1 / len(PHI[0])) * nSum
        beta = 1 / beta_inv
        if (abs(alpha - alphaT) < 0.001) and (abs(beta - betaT) < 0.001):
            break
        alphaT = alpha
        betaT = beta
    Sigma_NN = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))  # PRML 公式3.54
    mu_NN = beta * np.dot(Sigma_NN, np.dot(PHI.T, yMat))
    print("alpha = %f, beta = %f " % (alpha, beta))

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat, yMat, s=30, c='red', marker='s')
    x = np.arange(0, 13.0, 0.1)
    y = np.dot(mu_NN, phi(x))
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return mu_NN


def normal(xMat, yMat):
    PHI = []
    for x in xMat:
        PHI.append(phi(x))
    PHI = np.array(PHI)
    print("PHI = ", PHI)

    # Normal Linear Regression
    omega_normal = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, yMat))
    print("omega(normal linear Regression) = ", omega_normal)
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat, yMat, s=30, c='red', marker='s')
    x = np.arange(0, 13.0, 0.1)
    y = np.dot(omega_normal, phi(x))
    ax.plot(x, y, c='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return omega_normal


def testBayes(mu, xMat, yMat):
    PHI = []
    t = []
    for x in xMat:
        PHI.append(phi(x))
    PHI = np.array(PHI)
    sum = 0.0
    for i in range(len(PHI)):
        t.append(pow(np.dot(mu, PHI[i])-yMat[i], 2))
        sum += pow(np.dot(mu, PHI[i])-yMat[i], 2)
    print("Error t = ", t)
    print("RMSE = ", np.sqrt(sum/len(yMat)))
