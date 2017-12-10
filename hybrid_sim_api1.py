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
    
class ImmutableError(Exception):
    """ This error is raised when an attempt is made to modify a constant signal """
    def __init__(self, *args, **kwds):
        super(self).__init__(*args, **kwds)
        # that's all

class Signal():
    """ A N-dimensional NumPy array of data, with a fixed shape and dtype """
    def __init__(self, dShape, dtype1, const=False):
        self.dShape = dShape
        self.dtype1 = dtype1
        self.data = np.empty(dShape, dtype=dtype1)
        self.initialized = False
        self.isConstant = const
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
        except (TypeError, ValueError):
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
                if not signal1.isConstant:
                    try:
                        updated = signal1.updateData(data2)
                    except (TypeError, ValueError):
                        pass
                else:
                    raise ImmutableError
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
        self.initialized = False
        self.function1 = simFunction1
        self.functData = functData
        self.props = props
        self.ioPorts = {}
        self.ioProps = {}
        self.inputNames = None
        self.outputNames = None
        
    def init_outputs(self, outputNames, initSignals):
        success = False
        if not self.initialized:
            if len(outputNames) == self.nOutputs:
                outputInds = list(range(self.nOutputs))
                for name, sigObject, n in zip(outputNames, initSignals, outputInds):
                    if 'out' in self.ioPorts:
                        outputs1 = self.ioPorts['out']
                    else:
                        outputs1 = {}
                    if n in outputs1:
                        if outputs1[n][1] is not None:
                            pass
                        else:
                            outputs1[n] = (name, sigObject)
                    else:
                        outputs1[n] = (name, sigObject)
                success = True
            else:
                pass
        else:
            pass
        return success
                
        
    def connect(self, block2, linkPorts):
        success = False
        if self.initialized:
            for outN, inN in linkPorts:
                if outN < self.nOutputs and inN < block2.nInputs:
                    if 'out' in self.ioPorts:
                        outputs1 = self.ioPorts['out']
                        if outN in outputs1:
                            (output_name, outObject) = outputs1[outN]
                        else:
                            break
#                            output_name = 'output_{0:0>2d}'.format(outN)
#                            outObject = None
#                            outputs1[outN] = (output_name, outObject)
#                            self.ioPorts['out'] = outputs1
                    else:
                        break
#                        outputs1 = {}
#                        for n in self.nOutputs:
#                            output_name = 'output_{0:0>2d}'.format(n)
#                            outObject = None
#                            outputs1[n] = (output_name, outObject)
#                        self.ioPorts['out'] = outputs1
                    if 'in' in block2.ioPorts:
                        inputs1 = block2.ioPorts['in']
                    else:
                        inputs1 = {}
                    if inN in inputs1:
                        pass
                    else:
                        input_name = output_name
                        inObject = outObject
                        inputs1[inN] = (input_name, inObject)
                        block2.ioPorts['in'] = inputs1
                    success = True
                else:
                    pass
        else:
            pass
        return success
    
    
                        
                    
                    

