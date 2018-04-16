# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:01:53 2017

@author: Byron Burks
"""

# Simulation Framework

# hybrid_sim_api1.py

import numpy as np
from contextlib import ContextDecorator
#from Queue import Queue
#from threading import Thread, Event
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
        if type(signal2) is Signal:
            dSpecs2 = signal2.dimSpecs
            nd2 = signal2.ndim
            dShape2 = signal2.shape
            d1.update(dict(zip(dSpecs2, dShape2)))
            dtype2 = signal2.dtype
        elif np.ndim(signal2) == 0:
            dSpecs2 = ()
            nd2 = 0
            dShape2 = ()
            if type(signal2) in (float, bool, int):
                dtype2 = type(signal2)
            else:
                raise TypeError
        else:
            raise TypeError
        allSpecs.append(dSpecs2)
        allDims.append(nd2)
        allShapes.append(dShape2)
        allTypes.append(dtype2)
    lastVals = None
    for vals in zip(allSpecs, allDims, allTypes):
        if lastVals is not None:
            (dSpecs1, nd1, dtype1) = lastVals
        else:
            lastVals = vals
            continue
        dSpecs2, nd2, dtype2 = vals
        if nd1 == 0 and nd2 == 0:
            nd3 = 0
            dSpecs3b = ()
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
        for k3, (dSpecs1, signal1) in enumerate(zip(allSpecs, allSignals)):
            reps1 = allReps[k3]
            if k2 in dSpecs1:
                reps1.append(whole())
            else:
                reps1.append(None)
            allReps[k3] = reps1
        dShape3a.append(d1[k2])
    dShape3b = tuple(dShape3a)
    outSignals = []
    for reps1, dtype1, signal1 in zip(allReps, allTypes, allSignals):
        inds1 = createInds(nd3, dims, reps1)
        if signal1.isConstant:
            signalOut = createSignal(dShape3b, (), dtype1, const=True)
        else:
            signalOut = createSignal(dShape3b, dSpecs3b, dtype1)
        signalOut.initData(signal1, inds1)
        outSignals.append(signalOut)
    return tuple(outSignals)

def createInds(nd, *args):
    if len(args) > 1 and nd > 0:
        dims, reps = args[:3]
    elif nd > 0:
        return whole()
    else:
        return None
    iList = []
    for k in range(nd):
        if k in dims:
            k2 = dims.index(k)
            iList.append(reps[k2])
        else:
            iList.append(whole())
    if len(iList) > 0:
        return SigIndices(iList)
    else:
        return None

def createSignal(dShape1, dimSpecs1, dtype1, const=False, autoInit=False):
    if (not const and len(dShape1) == len(dimSpecs1)) or const:
        signal1 = Signal(dShape1, dtype=dtype1)
        signal1.setupSignal(dimSpecs1, const=const)
        if autoInit:
            signal1.initData(0, createInds(len(dShape1)))
        else:
            pass
        return signal1
    else:
        print('Mutable signals must have consistent dimension sizes; all dimensions must be specified.')
        raise TypeError

def createConst(cdata1, dtype1=float, dShape1=()):
    cdata2 = np.array(cdata1)
    const1 = Signal(dShape1, dtype=dtype1)
    const1.setupSignal((), const=True)
    const1.initData(cdata2, True)
    return const1

def cond2Integers(condition1):
    if type(condition1) is Signal:
        dShape1 = condition1.shape
        nd = condition1.ndim
        dims = list(range(nd))
    else:
        raise NotImplementedError
    if nd > 1:
        condition1b = condition1.flatten()
        allIndices = np.indices(dShape1)
        iList = []
        for k in dims:
            inds1 = allIndices[k]
            inds1b = inds1.flatten()
            inds2 = inds1b[condition1b]
            iList.append(inds2)
        condIndices = createInds(nd, dims, iList)
    elif nd == 1:
        condIndices = createInds(nd, [0], condition1)
    else:
        condIndices = None
    return condIndices

def ifcond(condition, allSignals):
    condition2, *allSignals2 = combineSignals(condition, *allSignals)
    condIndices = cond2Integers(condition2)
    return LogicBlock(condition2, condIndices, allSignals2)

def elifcond(ifobj, condition, allSignals):
    condition2, *allSignals2 = combineSignals(condition, *allSignals)
    condition3 = (condition2 & ~ifobj.logicalOut) & ifobj.logicalLast
    condIndices = cond2Integers(condition3)
    return LogicBlock(condition3, condIndices, allSignals2)

def elseclause(ifobj, allSignals):
    condition, *allSignals2 = combineSignals(ifobj.logicalOut, *allSignals)
    condition2 = ~condition & ifobj.logicalLast
    condIndices = cond2Integers(condition2)
    return LogicBlock(condition2, condIndices, allSignals2)

def nestedifcond(parentobj, condition, allSignals):
    condition2, *allSignals2 = combineSignals(condition, *allSignals)
    condition3 = condition2 & parentobj.logicalOut
    condIndices = cond2Integers(condition3)
    return LogicBlock(condition3, condIndices, allSignals2, lastExpr=parentobj.logicalOut)

def shiftSignal(signal1, N, padVals=None, dir1='right', axis1='0'):
    if dir1 == 'right':
        nd1 = np.ndim(signal1)
        if nd1 > 0:
            dShape1 = list(signal1.shape)
            dShape1[int(axis1)] = N
            dShape2 = tuple(dShape1)
            dims1 = list(range(nd1))
            reps1 = [None for k in dims1]
            reps1[int(axis1)] = whole()
            inds1 = createInds(nd1, dims1, reps1)
            inds2 = createInds(nd1, [int(axis1)], [slice(-N)])
        else:
            raise IndexError
        if padVals is None:
            padding1 = np.zeros(dShape2, dtype=signal1.dtype)
            padding2 = padding1
        else:
            padding2 = np.zeros(dShape2, dtype=signal1.dtype)
            padding1 = np.array(padVals, dtype=signal1.dtype)
            padding2[whole()] = padding1[inds1]
        data2 = np.r_[axis1, padding2, signal1[inds2]]
        signal2 = createSignal(signal1.shape, signal1.dimSpecs1, signal1.dtype)
        signal2.initData(0, whole())
        signal2.updateData(data2, whole())
    else:
        nd1 = np.ndim(signal1)
        if nd1 > 0:
            dShape1 = list(signal1.shape)
            dShape1[int(axis1)] = N
            dShape2 = tuple(dShape1)
            dims1 = list(range(nd1))
            reps1 = [None for k in dims1]
            reps1[int(axis1)] = whole()
            inds1 = createInds(nd1, dims1, reps1)
            inds2 = createInds(nd1, [int(axis1)], [slice(N, None)])
        else:
            raise IndexError
        if padVals is None:
            padding1 = np.zeros(dShape2, dtype=signal1.dtype)
            padding2 = padding1
        else:
            padding2 = np.zeros(dShape2, dtype=signal1.dtype)
            padding1 = np.array(padVals, dtype=signal1.dtype)
            padding2[whole()] = padding1[inds1]
        data2 = np.r_[axis1, signal1[inds2], padding2]
        signal2 = createSignal(signal1.shape, signal1.dimSpecs1, signal1.dtype)
        signal2.initData(0, whole())
        signal2.updateData(data2, whole())
    return signal2

def createBlock(name, nInputs, nOutputs, simFunction1, initData, dataMontitor, functKwds=None):
    block1 = SimBlock(name, nInputs, nOutputs, simFunction1, functKwds=functKwds)
    (outputNames, outputData) = initData
    block1.initOutputs(outputNames, outputData, dataMontitor)
    return block1

def createDX(uniqueId):
    x = IgnoreDim(uniqueId + 100)
    return x
        
def whole(stop=None):
    return slice(stop)

class IgnoreDim(int):
    """ Represents a dimension spec that is NOT connected to a simulation input """
    def __repr__(self):
        return 'x' + str(self - 100)
        
    def __hash__(self):
        return int(self)
    
    
class SigIndices(tuple):
    """ Modified indexing tuple """
    def __str__(self):
        str1 = ','.join(map(str, self))
        str2 = str1.replace('True', ':')
        return 'indices = {}'.format(str2)
    
    def __repr__(self):
        str1 = ','.join(map(str, self))
        str2 = str1.replace('True', ':')
        return 'indices = {}'.format(str2)
    
    
class ImmutableError(Exception):
    """ This error is raised when an attempt is made to modify a constant signal """


class Signal(np.ndarray):
    """ A N-dimensional NumPy array of data, with a fixed shape and dtype """
    def __init__(self, *args, **kwds):
        super().__init__()
        self.initialized = False
        self.finalOutput = False
        self.dimSpecs = ()
        self.ignoreDims = ()
        self.isConstant = False
        self.isEnableFlag = False
    
    def setupSignal(self, dimSpecs, const=False, ignoreDims=None, isEnableFlag=False):
        self.dimSpecs = dimSpecs
        self.isConstant = const
        if ignoreDims is not None:
            self.ignoreDims = ignoreDims
        else:
            pass
        if issubclass(self.dtype, int) and not const:
            self.isEnableFlag = isEnableFlag
        else:
            pass
        
    def initData(self, data2, inds):
        # data2 must have the same shape and a compatible dtype as self.data
        if self.initialized == False:
            nd1 = self.ndim
            nd2 = np.ndim(data2)
            if nd1 == 0:
                self[None] = data2
            elif self.isConstant:
                self[whole()] = data2
                self.initialized = True
            elif np.size(data2) == 1:
                self[inds] = data2
                self.initialized = True
            else:
                if nd1 == nd2:
                    self[inds] = data2
                    self.initialized = True
                else:
                    pass
        else:
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
                self[inds] = data2
                updated = True
            else:
                raise ImmutableError
        else:
            pass
        return updated
    
    def __repr__(self):
        return str(np.array(self))
    
    def __and__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.logical_and(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    
    def __or__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.logical_or(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    
    def __invert__(self):
        outSignal = createSignal(self.shape, self.dimSpecs, bool)
        outSignal.initData(np.logical_not(self), createInds(self.ndim))
        return outSignal
    
    def __eq__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.equal(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    
    def __ne__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.not_equal(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    
    def __lt__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.less(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    
    def __le__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.less_equal(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    
    def __gt__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.greater(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    
    def __ge__(self, signal2):
        signal3, signal4 = combineSignals(self, signal2)
        outSignal = createSignal(signal3.shape, signal3.dimSpecs, bool)
        outSignal.initData(np.greater_equal(signal3, signal4), createInds(signal3.ndim))
        return outSignal
    

class LogicalArray():
    """ Allows for if statements that operate on arrays """
    def __init__(self, logicExpr, logicIndices, allSignals, lastExpr=None):
        self.logicalOut = logicExpr
        if lastExpr is not None:
            self.logicalLast = lastExpr
        else:
            self.logicalLast = logicExpr | True
        self.logicalInds = logicIndices
        self.allSignals = allSignals
        
    def getAll(self):
        return (self.logicalInds, *self.allSignals)


class LogicBlock(LogicalArray, ContextDecorator):
    """ Context manager for logic blocks """
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        return False


class DataMonitor():
    """ Creates an object containing a set of relevant signals with unique names """
    def __init__(self, name, simQueue, outQueue):
        self.name = name
        self.simQueue = simQueue
        self.outQueue = outQueue
        self.data = {}
        self.sigConnections = {}
        
    def addSignals(self, names, sigObjects):
        for name, signal, cblocks in zip(names, sigObjects):
            if name not in self.data:
                self.data[name] = signal
            else:
                pass
            
    def addConnections(self, sigName, connectedBlocks):
        if sigName in self.sigConnections:
            cblocks1 = self.sigConnections[sigName]
            cblocks1.extend(connectedBlocks)
            self.sigConnections[sigName] = cblocks1
        else:
            self.sigConnections[sigName] = connectedBlocks
            
    def updateSignals(self, names, data_sets):
        for name, data2 in zip(names, data_sets):
            updated = False
            if name in self.data and allNotInvalid(data2):
                signal1 = self.data[name]
                inds1 = createInds(signal1.ndim)
                updated = signal1.updateData(data2, inds1)
            else:
                pass
            if updated:
                if name in self.sigConnections:
                    for block1 in self.sigConnections[name]:
                        self.simQueue.put((block1, self))
                elif signal1.finalOutput:
                    self.outQueue.put(signal1)
                else:
                    pass
            else:
                pass


class SimBlock():
    """ A single simulation entity encapsulating a function """
    def __init__(self, blockName, nInputs, nOutputs, simFunction1, functKwds=None):
        self.name = blockName
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.initValues = {}
        self.initialized = False
        self.function1 = simFunction1
        self.functData = {}
        if functKwds is not None:
            self.functKwds = functKwds
        else:
            self.functKwds = {}
        self.ioPorts = {}
        self.ioProps = {}
        self.inputNames = []
        self.outputNames = []
        self.enable = True
        
    def initOutputs(self, outputNames, initSignals, dataMonitor):
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
                self.initialized = True
                self.outputNames = outputNames
                dataMonitor.addSignals(outputNames, initSignals)
            else:
                pass
        else:
            pass
        return success
    
    def addFunctData(self, functData):
        self.functData.update(functData)
        
    def connect(self, block2, linkPorts, dataMonitor):
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
                        block2.inputNames.append(input_name)
                        dataMonitor.addConnections(output_name, [block2])
                        success = True
                else:
                    pass
        else:
            pass
        return success
    
    def runBlock(self, dataMonitor):
        isRunnable = True
        if self.initialized:
            inputSignals = []
            if len(self.inputNames) > 0:
                for k1, name in enumerate(self.inputNames):
                    inputs1 = self.ioPorts['in']
                    if k1 in inputs1:
                        if inputs1[k1].initialized:
                            inputSignals.append(dataMonitor.data[name])
                        else:
                            isRunnable = False
                            break
                    else:
                        isRunnable = False
                        break
            else:
                pass
            if isRunnable:
                outputs1 = self.function(*inputSignals, **self.functKwds)
                outputs2 = list(outputs1)
                dataMonitor.updateSignals(self.outputNames, outputs2)
            else:
                pass
        else:
            isRunnable = False
            # block not initialized!
            raise RuntimeError
        return isRunnable


class SimModel():
    """ Simulation model class """
    def __init__(self, name, simQueue, outQueue, domainUnits=None):
        self.simq = simQueue
        self.outq = outQueue
        self.name = name
        self.measUnits = domainUnits
        self.monitors = {}
        self.allBlocks = {}
        self.blockNames = []
        
    def resetModel(self):
        self.monitors = {}
        self.allBlocks = {}
        self.blockNames = []
    
    def buildModel(self, blockDeclarations, blockInitData, modelPortlist, monitorNames):
        'This method is automatically called by one or more module-level functions.'
        allMonitors = []
        for name in monitorNames:
            monitor1 = DataMonitor(name, self.simq, self.outq)
            allMonitors.append(monitor1)
        self.monitors = dict(zip(monitorNames, allMonitors))
        # create and initialize blocks
        for blockDecl, initData in zip(blockDeclarations, blockInitData):
            if len(blockDecl) > 5:
                (blockName, nInputs, nOutputs, simFunction1, monitorName, functKwds) = blockDecl[:5]
                blockName2 = '.'.join([monitorName, blockName])
            elif len(blockDecl) == 5:
                (blockName, nInputs, nOutputs, simFunction1, monitorName) = blockDecl
                blockName2 = '.'.join([monitorName, blockName])
                functKwds = {}
            else:
                continue
            if monitorName in self.monitors:
                dataMonitor1 = self.monitors[monitorName]
            else:
                continue
            block1 = createBlock(blockName2, nInputs, nOutputs, simFunction1, initData, dataMonitor1, functKwds)
            self.blockNames.append(blockName2)
            self.allBlocks[blockName2] = (block1, dataMonitor1)
        # make connections based on netlist
        for portSpec in modelPortlist:
            (sourceName, outN, sinkName, inN) = portSpec
            (sourceBlock, monitor1) = self.allBlocks[sourceName]
            sinkBlock = self.allBlocks[sinkName][0]
            sourceBlock.connect(sinkBlock, [(outN, inN)], sourceBlock, monitor1)
        # done building
    
    

