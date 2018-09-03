#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


class ShowPlt(object):
    def __init__(self, dataX, dataY, weights):
        self.dataX = dataX
        self.dataY = dataY
        self.weights = weights

    def plotMap(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.dataX)):
            if int(self.dataY[i]) == 2:
                ax.scatter(self.dataX[i, 0], self.dataX[i, 3], c='red')
            else:
                ax.scatter(self.dataX[i, 0], self.dataX[i, 3], c='blue')
        # ax.plot
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.show()
