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


    halo = [2,2]

    blocks = [blocking.getBlock(i) for i in range(blocking.numberOfBlocks)]
    blocksWithHalo = [blocking.getBlockWithHalo(i, halo) for i in range(blocking.numberOfBlocks)]


    assert_equals(blocks[0].begin,[0,0])
    assert_equals(blocks[0].end,  [3,3])


    print blocksWithHalo[0].outerBlock

    assert_equals(blocksWithHalo[0].innerBlock.begin,[0,0])
    assert_equals(blocksWithHalo[0].innerBlock.end,  [3,3])
    assert_equals(blocksWithHalo[0].outerBlock.begin,[0,0])
    assert_equals(blocksWithHalo[0].outerBlock.end,  [5,5])


    assert_equals(blocks[1].begin,[0,3])
    assert_equals(blocks[1].end,  [3,6])
    assert_equals(blocksWithHalo[1].innerBlock.begin,[0,3])
    assert_equals(blocksWithHalo[1].innerBlock.end,  [3,6])
    assert_equals(blocksWithHalo[1].outerBlock.begin,[0,1])
    assert_equals(blocksWithHalo[1].outerBlock.end,  [5,7])

    assert_equals(blocks[2].begin,[0,6])
    assert_equals(blocks[2].end,  [3,7])
    assert_equals(blocksWithHalo[2].innerBlock.begin,[0,6])
    assert_equals(blocksWithHalo[2].innerBlock.end,  [3,7])
    assert_equals(blocksWithHalo[2].outerBlock.begin,[0,4])
    assert_equals(blocksWithHalo[2].outerBlock.end,  [5,7])

    assert_equals(blocks[3].begin,[3,0])
    assert_equals(blocks[3].end,  [5,3])
    assert_equals(blocksWithHalo[3].innerBlock.begin,[3,0])
    assert_equals(blocksWithHalo[3].innerBlock.end,  [5,3])
    assert_equals(blocksWithHalo[3].outerBlock.begin,[1,0])
    assert_equals(blocksWithHalo[3].outerBlock.end,  [5,5])

    assert_equals(blocks[4].begin,[3,3])
    assert_equals(blocks[4].end,  [5,6])
    assert_equals(blocksWithHalo[4].innerBlock.begin,[3,3])
    assert_equals(blocksWithHalo[4].innerBlock.end,  [5,6])
    assert_equals(blocksWithHalo[4].outerBlock.begin,[1,1])
    assert_equals(blocksWithHalo[4].outerBlock.end,  [5,7])

    assert_equals(blocks[5].begin,[3,6])
    assert_equals(blocks[5].end,  [5,7])
    assert_equals(blocksWithHalo[5].innerBlock.begin,[3,6])
    assert_equals(blocksWithHalo[5].innerBlock.end,  [5,7])
    assert_equals(blocksWithHalo[5].outerBlock.begin,[1,4])
    assert_equals(blocksWithHalo[5].outerBlock.end,  [5,7])

