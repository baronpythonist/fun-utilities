# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:01:53 2017

@author: Byron Burks
"""

# Simulation Framework

# hybrid_sim_api1.py

import numpy as np
#from Queue import Queue
#import xlrd
#import pyqtgraph as pg


def allNotInvalid(data):
    if not any(np.isnan(data)):
        if all(np.logical_and(data < np.inf, data > -np.inf)):
            return True
        else:
            return False
    else:
        return False

class Signal():
    """ A N-dimensional NumPy array of data, with a fixed shape and dtype """
    def __init__(self, dShape, dtype1):
        self.dShape = dShape
        self.dtype1 = dtype1
        self.data = np.empty(dShape, dtype=dtype1)
        self.initialized = False
        self.connected_blocks = []
        
    def initData(self, data2):
        # data2 must have the same shape and a compatible dtype as self.data
        try:
            nd1 = self.data.ndim
            nd2 = data2.ndim
            if nd1 == nd2:
                if data2.shape == self.dShape:
                    inds = tuple([... for d in range(nd1)])
                    self.data[inds] = data2
                    self.initialized = True
                else:
                    pass
        except TypeError:
            pass
        
    def addAsBlockInput(self, block1):
        if block1 not in self.connected_blocks:
            self.connected_blocks.append(block1)
        else:
            pass
        
    def updateData(self, data2):
        updated = False
        if self.initialized:
            nd1 = self.data.ndim
            if data2.shape == self.dShape:
                inds = tuple([... for d in range(nd1)])
                self.data[inds] = data2
                updated = True
            else:
                pass
        else:
            pass
        return updated
        
    def getData(self):
        if self.initialized and allNotInvalid(self.data):
            return self.data
        else:
            raise ValueError


class Data_Monitor():
    """ Creates an object containing a set of relevant signals with unique names """
    def __init__(self, simQueue):
        self.simQueue = simQueue
        self.data = {}
        
    def addSignals(self, names, sigObjects):
        for name, signal in zip(names, sigObjects):
            if name not in self.data:
                self.data[name] = sigObjects
            else:
                pass
            
    def updateSignals(self, names, data_sets):
        for name, data2 in zip(names, data_sets):
            updated = False
            if name in self.data:
                signal1 = self.data[name]
                try:
                    updated = signal1.updateData(data2)
                except TypeError:
                    pass
            else:
                pass
            if updated:
                for block1 in signal1.connected_blocks:
                    self.simQueue.put(block1)
            else:
                pass


class Sim_Block():
    """ A single simulation entity encapsulating a function """
    def __init__(self, nInputs, nOutputs, simFunction1, functData, props=None):
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.initValues = {}
        self.function1 = simFunction1
        self.functData = functData
        self.props = props
        self.ioPorts = {}
        self.ioProps = {}
        self.inputNames = None
        self.outputNames = None
        
    def connect(self, block2, linkPorts):
        pass

