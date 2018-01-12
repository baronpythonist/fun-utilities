# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:01:53 2017

@author: Byron Burks
"""

# Simulation Framework

# hybrid_sim_api1.py

import numpy as np
#from contextlib import ContextDecorator
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
    
def combineSignals(signal1, *remSignals):
    allSpecs = []
    allDims = []
    allShapes = []
    allTypes = []
    allSpecs.append(signal1.dimSpecs)
    allDims.append(signal1.ndim)
    allShapes.append(signal1.shape)
    allTypes.append(signal1.dtype)
    d1 = dict(zip(signal1.dimSpecs, signal1.shape))
    allSignals = [signal1, *remSignals]
    for signal2 in remSignals:
        if isinstance(signal2, Signal):
            dSpecs2 = signal2.dimSpecs
            nd2 = signal2.ndim
            dShape2 = signal2.shape
            d1.update(dSpecs2, dShape2)
            dtype2 = signal2.dtype
        elif signal2.ndim == 0:
            dSpecs2 = []
            nd2 = 0
            dShape2 = ()
            if isinstance(signal2, float):
                dtype2 = float
            else:
                dtype2 = int
        else:
            raise TypeError
        allSpecs.append(dSpecs2)
        allDims.append(nd2)
        allShapes.append(dShape2)
        allTypes.append(dtype2)
    lastVals = None
    for vals in zip(allSpecs, allDims, allTypes):
        if lastVals is not None:
            dSpecs1, nd1, dtype1 = lastVals
        else:
            lastVals = vals
            continue
        dSpecs2, nd2, dtype2 = vals
        if nd1 == 0 and nd2 == 0:
            nd3 = 0
            dSpecs3b = []
        elif nd2 == 0:
            nd3 = nd1
            dSpecs3b = dSpecs1
        elif nd1 == 0:
            nd3 = nd2
            dSpecs3b = dSpecs2
        else:
            dSpecs3a = set(dSpecs1)
            dSpecs3a.union(dSpecs2)
            dSpecs3b = list(dSpecs3a)
            dSpecs3b.sort()
            nd3 = len(dSpecs3b)
        lastVals = (dSpecs3b, nd3, dtype2)
    (dSpecs3b, nd3, dtype2) = lastVals
    dims = list(range(nd3))
    allReps = [[] for s in allSpecs]
    dShape3a = []
    for k1, k2 in zip(dims, dSpecs3b):
        for k3, dSpecs1 in enumerate(allSpecs):
            reps1 = allReps[k3]
            if k2 in dSpecs1:
                reps1.append(...)
            else:
                reps1.append(None)
            allReps[k3] = reps1
        dShape3a.append(d1[k2])
    dShape3b = tuple(dShape3a)
    outSignals = []
    for reps1, dtype1, signal1 in zip(allReps, allTypes, allSignals):
        inds1 = SigIndices(nd3, dims, reps1)
        signalOut = Signal(dShape3b, dSpecs3b, dtype1)
        signalOut.initData(signal1, inds1)
        outSignals.append(signalOut)
    return tuple(outSignals)

    
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
        
    def initData(self, data2, inds):
        # data2 must have the same shape and a compatible dtype as self.data
        try:
            nd1 = self.ndim
            nd2 = data2.ndim
            if nd1 == nd2:
                if data2.shape == self.shape:
#                    inds = SigIndices(nd1)
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
        
    def updateData(self, data2, inds):
        updated = False
        if self.initialized:
#            nd1 = self.ndim
            if not self.isConstant:
                if data2.shape == self.dShape:
    #                inds = SigIndices(nd1)
                    self[inds] = data2
                    updated = True
                else:
                    pass
            raise ImmutableError
        else:
            pass
        return updated
    
    def __and__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.logical_and(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal
    
    def __or__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.logical_or(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal
    
    def __invert__(self):
        outSignal = Signal(self.shape, self.dimSpecs, bool)
        outSignal.initData(np.logical_not(self), SigIndices(self.ndim))
        return outSignal
    
    def __eq__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.equal(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal
    
    def __ne__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.not_equal(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal
    
    def __lt__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.less(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal
    
    def __le__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.less_equal(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal
    
    def __gt__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.greater(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal
    
    def __ge__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = Signal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.greater_equal(signal3, signal4), SigIndices(signal3.ndim))
        return outSignal

class LogicalArray():
    """ Allows for if statements that operate on arrays """
    def __init__(self, logicExpr, allSignals):
        self.logicalOut = logicExpr
        self.allSignals = allSignals
        
        
 
class LogicBlock(LogicalArray):
    """ Context manager for logic blocks """
    def __init__(self, logicExpr, allSignals):
        super(self).__init__(logicExpr, allSignals)
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        return False
        

def ifcond(condition, allSignals):
    condition2, *allSignals2 = combineSignals(condition, *allSignals)
    return LogicBlock(condition2, allSignals2)

def elifcond(ifobj, condition, allSignals):
    condition2, *allSignals2 = combineSignals(condition, *allSignals)
    condition3 = condition2 and not ifobj.logicalOut
    return LogicBlock(condition3, allSignals2)

def elseclause(ifobj, allSignals):
    condition, *allSignals2 = combineSignals(ifobj.logicalOut, *allSignals)
    return LogicBlock(not condition, allSignals2)

def nestedifcond(parentobj, condition, allSignals):
    condition2, *allSignals2 = combineSignals(condition, *allSignals)
    condition3 = condition2 and parentobj.logicalOut
    return LogicBlock(condition3, allSignals2)

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
                except (TypeError, ValueError):
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
                    else:
                        break
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

