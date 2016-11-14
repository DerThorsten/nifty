from _tools import *

__all__ = []

for key in _tools.__dict__.keys():
    __all__.append(key)





if False:
    class Block(object):
        def __init__(self,begin,end):
            self.begin = tuple(begin)
            self.end = tuple(end)

        @property
        def slicing(self):
            return [slice(b,e) for b,e in zip(self.begin, self.end)]

        @property
        def  shape(self):
            return tuple([e-b for e,b in zip(self.end, self.begin)])
        


        def __repr__(self):
            #return "(begin: %s end: %s)"%(str(self.begin),str(self.end))
            return "(%s - %s)"%(str(self.begin),str(self.end))

    class BlockWithHalo(object):
        def __init__(self, outerBlock, innerBlock):
            self.outerBlock = outerBlock
            self.innerBlock = innerBlock

            innerBlockShape = innerBlock.shape

            innerBlockLocalBegin = [ibb - obb for ibb,obb in zip(innerBlock.begin, outerBlock.begin)]
            innerBlockLocalEnd = [iblb + ibs for iblb, ibs in zip(innerBlockLocalBegin, innerBlockShape)]

            self.innerBlockLocal = Block(innerBlockLocalBegin, innerBlockLocalEnd)

        @property
        def  shape(self):
            return self.outerBlock.shape

        @property
        def  shape(self):
            return self.outerBlock.shape

        def __repr__(self):
            return "(outerBlock: %s innerBlock: %s)"\
                %(str(self.outerBlock),str(self.innerBlock))

    class Blocking3D(object):
        def __init__(self,roiBegin, roiEnd, blockShape, blockShift = (0,0,0)):
            assert len(roiBegin) == 3
            assert len(roiEnd) == 3
            assert len(blockShape) == 3

            for i in range(3):
                assert (blockShift[i] >= 0) and (blockShift[i] < blockShape[i])

            self.roiBegin = roiBegin
            self.roiEnd = roiEnd
            self.blockShape = blockShape
            self.roiShape = [rE-rB for rE,rB in zip(roiEnd, roiBegin)]
            self.blockShift = blockShift

            self.shiftedRoiBegin = [rb-bs for rb,bs in zip(roiBegin,blockShift)]

            withShiftShape = [rs + bs for rs,bs in zip(self.roiShape, blockShift)]


            # blocks per axis
            self.blocksPerAxis = [rs//bs + int(rs%bs !=0) for rs,bs in zip(withShiftShape, blockShape)]

            self.numberOfBlocks = self.blocksPerAxis[2]*self.blocksPerAxis[1]*self.blocksPerAxis[0]
            self.blockPerAxisStrides = [self.blocksPerAxis[2]*self.blocksPerAxis[1], self.blocksPerAxis[2], 1]

            numpy.zeros(shape=self.blocksPerAxis, dtype=numpy.object)


        def __len__(self):
            return self.numberOfBlocks




        def __blockCoordToBeginEnd(self, blockCoord):

            coordBegin =  [rb + bc*bs for bc,bs,rb in zip(blockCoord, self.blockShape, self.shiftedRoiBegin)]
            coordEnd =  [min(cb + bs, re) for cb,bs,re in zip(coordBegin, self.blockShape, self.roiEnd)]

            coordBegin = [max(rb,cb) for rb,cb in zip(self.roiBegin, coordBegin)]


            return coordBegin, coordEnd

        def __blockIndexToCoord(self, index):
            i = index
            x0 = i / self.blockPerAxisStrides[0]
            i = i - x0*self.blockPerAxisStrides[0]
            x1 = i /   self.blockPerAxisStrides[1]
            x2 = i - x1*self.blockPerAxisStrides[1]

            return (x0,x1,x2)


        def getBlock(self, index):
            if isinstance(index, (tuple, list)):
                begin,end = self.__blockCoordToBeginEnd(blockCoord)
                return Block(begin, end)
            else:
                blockCoord = self.__blockIndexToCoord(index)
                begin,end = self.__blockCoordToBeginEnd(blockCoord)
                return Block(begin, end)


        def getBlockWithHalo(self, index, halo):
            innerBlock = self.getBlock(index)
            outerBegin = [max(rb, ib - h) for rb,ib,h in zip(self.roiBegin, innerBlock.begin, halo)]
            outerEnd   = [min(re, ie + h) for re,ie,h in zip(self.roiEnd, innerBlock.end, halo)]

            outerBlock = Block(outerBegin, outerEnd)

            return BlockWithHalo(outerBlock=outerBlock, innerBlock=innerBlock)





def blocking(roiBegin, roiEnd, blockShape, blockShift=None):
    ndim = len(roiBegin)

    assert ndim == len(roiEnd)
    assert ndim == len(blockShape)
    if blockShift is not None:
        assert ndim == len(blockShift)
    else:
        blockShift = [0]*ndim


    if ndim == 2:
        blockingCls = Blocking2d
    elif ndim == 3:
        blockingCls = Blocking3d
    else:
        raise RuntimeError("only 2d and 3d blocking is implemented currently")
        
    return blockingCls(
        [int(v) for v in roiBegin],
        [int(v) for v in roiEnd],
        [int(v) for v in blockShape],
        [int(v) for v in blockShift]
    )



