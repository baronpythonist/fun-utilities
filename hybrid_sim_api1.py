# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:01:53 2017

@author: Byron Burks
"""

# Simulation Framework

# hybrid_sim_api1.py

import numpy as np
from contextlib import contextmanager
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
    
def combineSignals(signal1, signal2):
    dSpecs1 = signal1.dimSpecs
    nd1 = signal1.ndim
    dShape1 = signal1.shape
    if isinstance(signal2, Signal):
        dSpecs2 = signal2.dimSpecs
        nd2 = signal2.ndim
        dShape2 = signal2.shape
    else:
        dSpecs2 = []
        nd2 = 0
        dShape2 = ()
    if nd1 == 0 and nd2 == 0:
        nd3 = 0
        dSpecs3b = []
        dShape3 = ()
    elif nd2 == 0:
        nd3 = nd1
        dSpecs3b = dSpecs1
        dShape3 = dShape1
    elif nd1 == 0:
        nd3 = nd2
        dSpecs3b = dSpecs2
        dShape3 = dShape2
    else:
        dSpecs3a = set(dSpecs1)
        dSpecs3a.union(dSpecs2)
        dSpecs3b = list(dSpecs3a)
        dSpecs3b.sort()
        nd3 = len(dSpecs3b)
        
    dims = list(range(nd3))
    reps1 = []
    reps2 = []
    for k1, k2 in zip(dims, dSpecs3b):
        if k2 in dSpecs1:
            reps1.append(...)
        else:
            reps1.append(None)
        if k2 in dSpecs2:
            reps2.append(...)
        else:
            reps2.append(None)
    inds1 = SigIndices(nd3, dims, reps1)
    inds2 = SigIndices(nd3, dims, reps2)
    

    
class SigIndices(tuple):
    """ Modified indexing tuple """
    def __init__(self, nd, *args):
        if len(args) > 1:
            dims, reps = args[:2]
        else:
            dims, reps = [], []
        iList = []
        for k in range(nd):
            if k in dims:
                k2 = dims.index(k)
                iList.append(reps[k2])
            else:
                iList.append(...)
        if len(iList) == 0:
            iList = None
        else:
            pass
        super(self).__init__(iList)
        
    def __str__(self):
        str1 = ','.join(str(self))
        str2 = str1.replace('Ellipsis', ':')
        return 'indices = {}'.format(str2)
    
class ImmutableError(Exception):
    """ This error is raised when an attempt is made to modify a constant signal """
    def __init__(self, *args, **kwds):
        super(self).__init__(*args, **kwds)
        # that's all

class Signal(np.ndarray):
    """ A N-dimensional NumPy array of data, with a fixed shape and dtype """
    def __init__(self, dShape, dimSpecs, dtype1, const=False):
        super(self).__init__(dShape, dtype=dtype1)
        self.initialized = False
        self.dimSpecs = dimSpecs
        self.isConstant = const
        self.connected_blocks = []
        
    def initData(self, data2):
        # data2 must have the same shape and a compatible dtype as self.data
        try:
            nd1 = self.ndim
            nd2 = data2.ndim
            if nd1 == nd2:
                if data2.shape == self.shape:
                    inds = SigIndices(nd1)
                    self[inds] = data2
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
                inds = SigIndices(nd1)
                self.data[inds] = data2
                updated = True
            else:
                pass
        else:
            pass
        return updated
    
    def __and__(self, signal2):
        
        
        
#    def getData(self):
#        if self.initialized and allNotInvalid(self.data):
#            return self.data
#        else:
#            raise ValueError

class LogicalArray(contextmanager):
    """ Allows for if statements that operate on arrays """
    def __init__(self, logicExpr):
        pass


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
    
    
                        
                    
                    

