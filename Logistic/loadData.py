#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-


class DataLoader(object):
    def __init__(self, filename):
        self.filename = filename
        self.dataX = []
        self.dataY = []

    def loadData(self):
        f = open(self.filename, 'r+', encoding='utf-8')
        for line in f.readlines():
            lineArr = line.encode('utf-8').decode('utf-8-sig').strip().split(',')
            self.dataX.append(lineArr[:-1])
            self.dataY.append(lineArr[-1])
        return self.dataX, self.dataY
