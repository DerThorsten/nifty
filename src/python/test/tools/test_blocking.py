import unittest
import nifty.tools as nt


class TestBlocking(unittest.TestCase):

    def testBlockingBlocksPerAxis(self):

        roiBegin =   [0,0,0]
        roiEnd =     [5,6,7]
        blockShape = [3,3,3]
        blockShift = [0,0,0]

        blocking = nt.blocking(roiBegin=roiBegin,roiEnd=roiEnd,
                                 blockShape=blockShape, blockShift=blockShift)

        blocksPerAxis = blocking.blocksPerAxis

        self.assertEqual(blocksPerAxis[0],2)
        self.assertEqual(blocksPerAxis[1],2)
        self.assertEqual(blocksPerAxis[2],3)

        roiBegin =   [0,0,0]
        roiEnd =     [5,6,7]
        blockShape = [3,3,3]
        blockShift = [0,1,0]

        blocking = nt.blocking(roiBegin=roiBegin,roiEnd=roiEnd,
                                 blockShape=blockShape, blockShift=blockShift)


        blocksPerAxis = blocking.blocksPerAxis

        self.assertEqual(blocksPerAxis[0],2)
        self.assertEqual(blocksPerAxis[1],3)
        self.assertEqual(blocksPerAxis[2],3)

    def testBlocking2d(self):

        roiBegin =   [0,0]
        roiEnd =     [5,7]
        blockShape = [3,3]
        blockShift = [0,0]

        blocking = nt.blocking(roiBegin=roiBegin,roiEnd=roiEnd,
                                 blockShape=blockShape, blockShift=blockShift)

        blocksPerAxis = blocking.blocksPerAxis

        self.assertEqual(blocksPerAxis[0],2)
        self.assertEqual(blocksPerAxis[1],3)

        halo = [2,2]

        blocks = [blocking.getBlock(i) for i in range(blocking.numberOfBlocks)]
        blocksWithHalo = [blocking.getBlockWithHalo(i, halo) for i in range(blocking.numberOfBlocks)]

        self.assertEqual(blocks[0].begin,[0,0])
        self.assertEqual(blocks[0].end,  [3,3])

        self.assertEqual(blocksWithHalo[0].innerBlock.begin,[0,0])
        self.assertEqual(blocksWithHalo[0].innerBlock.end,  [3,3])
        self.assertEqual(blocksWithHalo[0].outerBlock.begin,[0,0])
        self.assertEqual(blocksWithHalo[0].outerBlock.end,  [5,5])

        self.assertEqual(blocks[1].begin,[0,3])
        self.assertEqual(blocks[1].end,  [3,6])
        self.assertEqual(blocksWithHalo[1].innerBlock.begin,[0,3])
        self.assertEqual(blocksWithHalo[1].innerBlock.end,  [3,6])
        self.assertEqual(blocksWithHalo[1].outerBlock.begin,[0,1])
        self.assertEqual(blocksWithHalo[1].outerBlock.end,  [5,7])

        self.assertEqual(blocks[2].begin,[0,6])
        self.assertEqual(blocks[2].end,  [3,7])
        self.assertEqual(blocksWithHalo[2].innerBlock.begin,[0,6])
        self.assertEqual(blocksWithHalo[2].innerBlock.end,  [3,7])
        self.assertEqual(blocksWithHalo[2].outerBlock.begin,[0,4])
        self.assertEqual(blocksWithHalo[2].outerBlock.end,  [5,7])

        self.assertEqual(blocks[3].begin,[3,0])
        self.assertEqual(blocks[3].end,  [5,3])
        self.assertEqual(blocksWithHalo[3].innerBlock.begin,[3,0])
        self.assertEqual(blocksWithHalo[3].innerBlock.end,  [5,3])
        self.assertEqual(blocksWithHalo[3].outerBlock.begin,[1,0])
        self.assertEqual(blocksWithHalo[3].outerBlock.end,  [5,5])

        self.assertEqual(blocks[4].begin,[3,3])
        self.assertEqual(blocks[4].end,  [5,6])
        self.assertEqual(blocksWithHalo[4].innerBlock.begin,[3,3])
        self.assertEqual(blocksWithHalo[4].innerBlock.end,  [5,6])
        self.assertEqual(blocksWithHalo[4].outerBlock.begin,[1,1])
        self.assertEqual(blocksWithHalo[4].outerBlock.end,  [5,7])

        self.assertEqual(blocks[5].begin,[3,6])
        self.assertEqual(blocks[5].end,  [5,7])
        self.assertEqual(blocksWithHalo[5].innerBlock.begin,[3,6])
        self.assertEqual(blocksWithHalo[5].innerBlock.end,  [5,7])
        self.assertEqual(blocksWithHalo[5].outerBlock.begin,[1,4])
        self.assertEqual(blocksWithHalo[5].outerBlock.end,  [5,7])

    def testNeighbors(self):
        blocking = nt.blocking(roiBegin=[0, 0, 0],
                               roiEnd=[10, 10, 10],
                               blockShape=[5, 5, 5])
        neighbors = {}
        for block_id in range(blocking.numberOfBlocks):
            nbrs = [blocking.getNeighborId(block_id,
                                           axis=i // 2,
                                           lower=bool(i % 2)) for i in range(6)]
            neighbors[block_id] = [nbr for nbr in nbrs if nbr != -1]
        for block_id in range(blocking.numberOfBlocks):
            nbrs = neighbors[block_id]
            for nbr_id in nbrs:
                self.assertTrue(block_id in neighbors[nbr_id])


if __name__ == '__main__':
    unittest.main()
