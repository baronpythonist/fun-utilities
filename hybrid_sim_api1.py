# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:01:53 2017

@author: Byron Burks
"""

# Simulation Framework

# hybrid_sim_api1.py

import numpy as np
import scipy.interpolate as snt
import scipy.signal as sig
from contextlib import ContextDecorator
#from Queue import Queue
#from threading import Thread, Event
#import xlrd
#import pyqtgraph as pg

def allNotInvalid(data):
    data2 = np.array(data.flatten())
    if not any(np.isnan(data2)):
        if np.max(data2) < np.inf and np.min(data2) > -np.inf:
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
    oddSpecs = [createDX(n) for n in range(10, 20)]
    for signal2 in remSignals:
        if type(signal2) is Signal:
            dSpecs2 = signal2.dimSpecs
            nd2 = signal2.ndim
            dShape2 = signal2.shape
            d1.update(dict(zip(dSpecs2, dShape2)))
            dtype2 = signal2.dtype
        else:
            try:
                dtype2 = signal2.dtype
                nd2 = np.ndim(signal2)
                dSpecs2 = tuple(oddSpecs[:nd2])
                dShape2 = np.shape(signal2)
            except AttributeError:
                if type(signal2) in (float, bool, int):
                    dtype2 = type(signal2)
                else:
                    raise TypeError
                if np.ndim(signal2) == 0:
                    dSpecs2 = ()
                    nd2 = 0
                    dShape2 = ()
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
    allReps = [[None for d in dims] for s in allSpecs]
    dShape3a = []
    for k1, k2 in zip(dims, dSpecs3b):
        for k3, (dSpecs1, signal1) in enumerate(zip(allSpecs, allSignals)):
            reps1 = allReps[k3]
            if k2 in dSpecs1:
                reps1[k1] = whole()
            else:
                reps1[k1] = None
            allReps[k3] = reps1
        dShape3a.append(d1[k2])
    dShape3b = tuple(dShape3a)
    outSignals = []
    for reps1, dtype1, signal1 in zip(allReps, allTypes, allSignals):
        inds1 = createInds(nd3)
        inds2 = createInds(nd3, dims, reps1)
        signalOut = createSignal(dShape3b, dSpecs3b, dtype1)
        try:
            signalOut.initData(signal1[inds2], inds1)
        except TypeError:
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
        
def copySignal(signalIn, includeValues=True):
    signalOut = createSignal(signalIn.shape, signalIn.dimSpecs, 
                             signalIn.dtype, const=signalIn.isConstant)
    inds1 = createInds(signalIn.ndim)
    if includeValues:
        signalOut.initData(signalIn, inds1)
    else:
        signalOut.initData(0, inds1)
    return signalOut

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

def sampleHold(data1, sampleLocations, axis1=0):
    'Samples "data1" at each integer index in "sampleLocations", holding the last sampled value'
    data2 = np.squeeze(data1)
    sampleLocations1b = np.squeeze(sampleLocations)
    lastDataN = np.size(data2, axis=axis1) - 1
    if np.size(sampleLocations1b) > 1:
        lastSampleN = np.max(sampleLocations1b, axis=axis1) - 1
        dims2 = list(range(data2.ndim))
        del dims2[axis1]
        dShape2 = list(data2.shape)
        dShape2[axis1] = 1
        if int(np.min(lastSampleN)) < lastDataN:
            if len(dims2) > 0:
                lastSamples = lastDataN*np.ones(tuple(dShape2), dtype=int)
            else:
                lastSamples = [lastDataN]
            sampleLocations2 = np.r_[str(axis1), sampleLocations1b, lastSamples]
            x1 = np.squeeze(sampleLocations2)
            inds1 = createInds(data2.ndim, [axis1], [x1])
        else:
            x1 = sampleLocations1b
            inds1 = createInds(data2.ndim, [axis1], [x1])
        y1 = np.squeeze(data2[inds1])
        sN = 4000
        x2 = np.indices(data2.shape)[axis1]
        if np.size(x1, axis=axis1) > sN:
            nSects = np.size(x1, axis=axis1)//sN + 1
            last_n = 0
            y2 = np.zeros(data2.shape)
            for n in range(nSects):
                n1 = last_n
                n2 = sN*(n+1) + 1
                if n2 >= np.size(x1, axis=axis1):
                    sliceN = slice(n1, None)
                    k1 = x1[n1]
                    k2 = x1[-1]
                    sliceK = slice(k1, k2)
                else:
                    sliceN = slice(n1, n2)
                    k1 = x1[n1]
                    k2 = x1[n2]
                    sliceK = slice(k1, k2)
                zf = snt.interp1d(x1[sliceN], y1[sliceN], kind='zero', axis=axis1)
                inds1b = createInds(x2.ndim, [axis1], [sliceK])
                y2[inds1b] = zf(x2[inds1b])
                last_n = n2
        else:
#            reps2 = [0 for d in dims2]
            inds2 = createInds(x2.ndim)
            zf = snt.interp1d(x1, y1, kind='zero', axis=axis1)
            try:
                y2 = zf(x2[inds2])
            except ValueError:
                print('Maximum input index: {n}'.format(n=np.max(x2[inds2])))
                print('\nMaximum index supported: {n}'.format(n=np.max(x1)))
                raise
        return y2
    else:
        return data2

def filterSegments(b, a, inputSignal, outputSignal, inputInds=None, outputInds=None, enabledRegions=True):
    segments1 = createSignal(b.shape[:1], b.dimSpecs[:1], bool, autoInit=True)
    segments2 = createSignal(a.shape[:1], a.dimSpecs[:1], bool, autoInit=True)
    inds1 = createInds(1)
    segments1.updateData(np.r_[1, np.sum(np.diff(b, axis=0), axis=1)], inds1)
    segments2.updateData(np.r_[1, np.sum(np.diff(a, axis=0), axis=1)], inds1)
    segInds1 = createSignal(segments1.shape, segments1.dimSpecs, np.uint32)
    segInds1.initData(np.indices(segments1.shape)[0], inds1)
    segInds2 = segInds1[(segments1 | segments2) & enabledRegions]
    segInds3 = segInds1[enabledRegions]
    interSignal = copySignal(inputSignal, includeValues=False)
    for si in segInds2:
        num1 = b[si,:]
        den1 = a[si,:]
        if si > 0:
            inds2 = createInds(inputSignal.ndim, [0], [si-1])
            initData = interSignal[inds2]
        else:
            inds2 = createInds(inputSignal.ndim, [0], [0])
            initData = inputSignal[inds2]
        if np.size(initData) == 1:
            initData2 = [initData]
        else:
            initData2 = initData
        inds3 = createInds(inputSignal.ndim, [0], [slice(si, None)])
        if np.size(inputSignal[inds3], axis=0) > 20:
            inds4 = createInds(inputSignal.ndim, [0], [segInds1 >= si])
            interSignal.updateData(sig.lfilter(num1, den1, inputSignal[inds3], 
                                               axis=0, zi=initData2), inds4)
        else:
            inds4 = createInds(inputSignal.ndim, [0], [segInds1 >= si])
            interSignal.updateData(initData2, inds4)
    if outputInds is not None:
        inds1b = outputInds
    else:
        inds1b = inds1
    outputSignal.updateData(sampleHold(interSignal, segInds3), inds1b)
#    return outputSignal

def createBlock(name, nInputs, nOutputs, simFunction1, initData, dataMonitor, functKwds=None):
    block1 = SimBlock(name, nInputs, nOutputs, simFunction1, functKwds=functKwds)
    (outputNames, outputData) = initData
    block1.initOutputs(outputNames, outputData, dataMonitor)
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
                if np.size(inds) > 0 and np.size(data2) > 0:
                    self[inds] = data2
                else:
                    pass
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
    
    def __getitem__(self, key):
        if np.size(key) > 0:
            return self[key]
        else:
            return []
    

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
    
    

