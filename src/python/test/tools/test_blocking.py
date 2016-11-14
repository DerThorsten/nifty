import nifty.tools as nt


from nose.tools import assert_equals


def testBlockingBlocksPerAxis():

    roiBegin =   [0,0,0]
    roiEnd =     [5,6,7]
    blockShape = [3,3,3]
    blockShift = [0,0,0]

    blocking = nt.blocking(roiBegin=roiBegin,roiEnd=roiEnd,
                             blockShape=blockShape, blockShift=blockShift)


    blocksPerAxis = blocking.blocksPerAxis

    assert_equals(blocksPerAxis[0],2)
    assert_equals(blocksPerAxis[1],2)
    assert_equals(blocksPerAxis[2],3)



    roiBegin =   [0,0,0]
    roiEnd =     [5,6,7]
    blockShape = [3,3,3]
    blockShift = [0,1,0]

    blocking = nt.blocking(roiBegin=roiBegin,roiEnd=roiEnd,
                             blockShape=blockShape, blockShift=blockShift)


    blocksPerAxis = blocking.blocksPerAxis

    assert_equals(blocksPerAxis[0],2)
    assert_equals(blocksPerAxis[1],3)
    assert_equals(blocksPerAxis[2],3)



def testBlocking2d():

    roiBegin =   [0,0]
    roiEnd =     [5,7]
    blockShape = [3,3]
    blockShift = [0,0]

    blocking = nt.blocking(roiBegin=roiBegin,roiEnd=roiEnd,
                             blockShape=blockShape, blockShift=blockShift)

    blocksPerAxis = blocking.blocksPerAxis


    assert_equals(blocksPerAxis[0],2)
    assert_equals(blocksPerAxis[1],3)


    blocks = [blocking.getBlock(i) for i in range(blocking.numberOfBlocks)]


    assert_equals(blocks[0].begin,[0,0])
    assert_equals(blocks[0].end,  [3,3])

    assert_equals(blocks[1].begin,[0,3])
    assert_equals(blocks[1].end,  [3,6])

    assert_equals(blocks[2].begin,[0,6])
    assert_equals(blocks[2].end,  [3,7])


    assert_equals(blocks[3].begin,[3,0])
    assert_equals(blocks[3].end,  [5,3])

    assert_equals(blocks[4].begin,[3,3])
    assert_equals(blocks[4].end,  [5,6])

    assert_equals(blocks[5].begin,[3,6])
    assert_equals(blocks[5].end,  [5,7])
