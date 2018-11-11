import nifty.cgp as ncgp
import nifty.graph.rag as nrag
import unittest

import nifty
import unittest
import nifty.cgp as ncgp
import numpy

numpy.random.seed(42)


class TestCgp2d(unittest.TestCase):

    def test_corner_case_3x3_grid_a(self):

        assertEq = self.assertEqual

        # 4 one cells are active
        # but still  no junction
        seg = [
            [1,1,2],
            [1,3,1],
            [1,1,1]
        ]
        seg = numpy.array(seg,dtype='uint32')
        tGrid = ncgp.TopologicalGrid2D(seg)

        numberOfCells  = tGrid.numberOfCells
        assertEq(numberOfCells,[0,2,3])

        tShape = tGrid.topologicalGridShape
        assertEq(tShape, [5,5])

        shape = tGrid.shape
        assertEq(shape, [3,3])


        # check the bounds
        bounds = tGrid.extractCellsBounds()

        bounds0 = bounds[0]
        bounds1 = bounds[1]
        boundedBy1 = bounds0.reverseMapping()
        boundedBy2 = bounds1.reverseMapping()

        assertEq(len(bounds0),0)
        assertEq(len(bounds1),2)
        assertEq(len(boundedBy1),2)
        assertEq(len(boundedBy2),3)


        # check the geometry
        geometryFS = tGrid.extractCellsGeometry(fill=True, sort1Cells=True)
        geometryF  = tGrid.extractCellsGeometry(fill=True, sort1Cells=False)
        geometryS  = tGrid.extractCellsGeometry(fill=False, sort1Cells=True)
        geometry   = tGrid.extractCellsGeometry(fill=False, sort1Cells=False)
        geos = [geometryFS,geometryF,geometryS,geometry]
        for geo in geos:

            for c in [0,1,2]:
               g = geo[c]
               assert len(g) == numberOfCells[c]

    def test_corner_case_3x3_grid_b(self):

        assertEq = self.assertEqual


        seg = [
            [1,1,1],
            [1,2,1],
            [1,1,1]
        ]

        seg = numpy.array(seg,dtype='uint32')
        tGrid = ncgp.TopologicalGrid2D(seg)

        numberOfCells  = tGrid.numberOfCells
        assertEq(numberOfCells,[0,1,2])

        tShape = tGrid.topologicalGridShape
        assertEq(tShape, [5,5])

        shape = tGrid.shape
        assertEq(shape, [3,3])


        # check the bounds
        bounds = tGrid.extractCellsBounds()

        bounds0 = bounds[0]
        bounds1 = bounds[1]
        boundedBy1 = bounds0.reverseMapping()
        boundedBy2 = bounds1.reverseMapping()

        assertEq(len(bounds0),0)
        assertEq(len(bounds1),1)
        assertEq(len(boundedBy1),1)
        assertEq(len(boundedBy2),2)


        # check the geometry
        geometryFS = tGrid.extractCellsGeometry(fill=True, sort1Cells=True)
        geometryF  = tGrid.extractCellsGeometry(fill=True, sort1Cells=False)
        geometryS  = tGrid.extractCellsGeometry(fill=False, sort1Cells=True)
        geometry   = tGrid.extractCellsGeometry(fill=False, sort1Cells=False)


        geos = [geometryFS,geometryF,geometryS,geometry]
        for geo in geos:

            for c in [0,1,2]:
               g = geo[c]
               assert len(g) == numberOfCells[c]

    def test_corner_case_3x3_grid_c(self):

        assertEq = self.assertEqual


        seg = [
            [1,1,3],
            [1,2,3],
            [1,1,3]
        ]

        seg = numpy.array(seg,dtype='uint32')
        tGrid = ncgp.TopologicalGrid2D(seg)

        numberOfCells  = tGrid.numberOfCells
        assertEq(numberOfCells,[2,4,3])

        tShape = tGrid.topologicalGridShape
        assertEq(tShape, [5,5])

        shape = tGrid.shape
        assertEq(shape, [3,3])


        # check the bounds
        bounds = tGrid.extractCellsBounds()

        bounds0 = bounds[0]
        bounds1 = bounds[1]
        boundedBy1 = bounds0.reverseMapping()
        boundedBy2 = bounds1.reverseMapping()

        assertEq(len(bounds0),2)
        assertEq(len(bounds1),4)
        assertEq(len(boundedBy1),4)
        assertEq(len(boundedBy2),3)


        # check the geometry
        geometryFS = tGrid.extractCellsGeometry(fill=True, sort1Cells=True)
        geometryF  = tGrid.extractCellsGeometry(fill=True, sort1Cells=False)
        geometryS  = tGrid.extractCellsGeometry(fill=False, sort1Cells=True)
        geometry   = tGrid.extractCellsGeometry(fill=False, sort1Cells=False)


        geos = [geometryFS,geometryF,geometryS,geometry]
        for geo in geos:

            for c in [0,1,2]:
               g = geo[c]
               assert len(g) == numberOfCells[c]

    def test_corner_case_3x3_grid_d(self):

        assertEq = self.assertEqual


        seg = [
            [1,1,1],
            [1,2,1],
            [1,1,3]
        ]

        #     01234
        #   --------------------
        # 0  |1|1|1| 0
        # 1  |-*-*-| 1
        # 2  |1|2|1| 2
        # 3  |-*-*-| 3
        # 4  |1|1|3| 4
        # ----------------------
        #     01234

        seg = numpy.array(seg,dtype='uint32').T
        tGrid = ncgp.TopologicalGrid2D(seg)

        numberOfCells  = tGrid.numberOfCells
        assertEq(numberOfCells,[0,2,3])

        tShape = tGrid.topologicalGridShape
        assertEq(tShape, [5,5])

        shape = tGrid.shape
        assertEq(shape, [3,3])


        # check the bounds
        bounds = tGrid.extractCellsBounds()

        bounds0 = bounds[0]
        bounds1 = bounds[1]
        boundedBy1 = bounds0.reverseMapping()
        boundedBy2 = bounds1.reverseMapping()

        assertEq(len(bounds0),0)
        assertEq(len(bounds1),2)
        assertEq(len(boundedBy1),2)
        assertEq(len(boundedBy2),3)


        # check the geometry
        #print(seg)
        geometryFS = tGrid.extractCellsGeometry(fill=True, sort1Cells=True)



        geometryF  = tGrid.extractCellsGeometry(fill=True, sort1Cells=False)
        geometryS  = tGrid.extractCellsGeometry(fill=False, sort1Cells=True)
        geometry   = tGrid.extractCellsGeometry(fill=False, sort1Cells=False)
        geos = [geometryFS,geometryF,geometryS,geometry]
        for geo in geos:

            for c in [0,1,2]:
               g = geo[c]
               assert len(g) == numberOfCells[c]


    def test_randomized_big(self):

        for x in  range(100):
            assertEq = self.assertEqual
            shape = (10, 20)
            size = shape[0]*shape[1]
            labels = numpy.random.randint(0, 4,size=size).reshape(shape)

            #print(labels)

            gg = nifty.graph.undirectedGridGraph(shape)
            cc = nifty.graph.connectedComponentsFromNodeLabels(gg, labels.ravel())
            cc = cc.reshape(shape) + 1
            cc = numpy.require(cc, dtype='uint32')

            tGrid = ncgp.TopologicalGrid2D(cc)
            numberOfCells  = tGrid.numberOfCells

            assertEq(numberOfCells[2], cc.max())


            # check the bounds
            bounds = tGrid.extractCellsBounds()
            boundedBy = {
                1:bounds[0].reverseMapping(),
                2:bounds[1].reverseMapping(),
            }
            try:
                # check the geometry
                geometryFS = tGrid.extractCellsGeometry(fill=True, sort1Cells=True)
                geometryF  = tGrid.extractCellsGeometry(fill=True, sort1Cells=False)
                geometryS  = tGrid.extractCellsGeometry(fill=False, sort1Cells=True)
                geometry   = tGrid.extractCellsGeometry(fill=False, sort1Cells=False)
                geos = [geometryFS,geometryF,geometryS,geometry]
            except:
                print(cc)
                import sys
                sys.exit()

    def test_randomized_medium(self):

        for x in  range(1000):
            assertEq = self.assertEqual
            shape = (7, 7)
            size = shape[0]*shape[1]
            labels = numpy.random.randint(0, 4,size=size).reshape(shape)

            #print(labels)

            gg = nifty.graph.undirectedGridGraph(shape)
            cc = nifty.graph.connectedComponentsFromNodeLabels(gg, labels.ravel())
            cc = cc.reshape(shape) + 1
            cc = numpy.require(cc, dtype='uint32')

            tGrid = ncgp.TopologicalGrid2D(cc)
            numberOfCells  = tGrid.numberOfCells

            assertEq(numberOfCells[2], cc.max())


            # check the bounds
            bounds = tGrid.extractCellsBounds()
            boundedBy = {
                1:bounds[0].reverseMapping(),
                2:bounds[1].reverseMapping(),
            }
            try:
                # check the geometry
                geometryFS = tGrid.extractCellsGeometry(fill=True, sort1Cells=True)
                geometryF  = tGrid.extractCellsGeometry(fill=True, sort1Cells=False)
                geometryS  = tGrid.extractCellsGeometry(fill=False, sort1Cells=True)
                geometry   = tGrid.extractCellsGeometry(fill=False, sort1Cells=False)
                geos = [geometryFS,geometryF,geometryS,geometry]
            except:
                print(cc)
                import sys
                sys.exit()

    def test_randomized_small(self):

        for x in  range(3000):
            assertEq = self.assertEqual
            shape = (4, 3)
            size = shape[0]*shape[1]
            labels = numpy.random.randint(1, 5,size=size).reshape(shape)

            #print(labels)

            gg = nifty.graph.undirectedGridGraph(shape)
            cc = nifty.graph.connectedComponentsFromNodeLabels(gg, labels.ravel())
            cc = cc.reshape(shape) + 1
            cc = numpy.require(cc, dtype='uint32')

            tGrid = ncgp.TopologicalGrid2D(cc)
            numberOfCells  = tGrid.numberOfCells

            assertEq(numberOfCells[2], cc.max())


            # check the bounds
            bounds = tGrid.extractCellsBounds()
            boundedBy = {
                1:bounds[0].reverseMapping(),
                2:bounds[1].reverseMapping(),
            }
            try:
                # check the geometry
                geometryFS = tGrid.extractCellsGeometry(fill=True, sort1Cells=True)
                geometryF  = tGrid.extractCellsGeometry(fill=True, sort1Cells=False)
                geometryS  = tGrid.extractCellsGeometry(fill=False, sort1Cells=True)
                geometry   = tGrid.extractCellsGeometry(fill=False, sort1Cells=False)
                geos = [geometryFS,geometryF,geometryS,geometry]
            except:
                print(cc)
                print("labels")
                print(labels)
                import sys
                sys.exit()
