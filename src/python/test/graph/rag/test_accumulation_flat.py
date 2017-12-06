import unittest
import numpy
import os
from shutil import rmtree
import zipfile
from subprocess import call
import nifty.graph.rag as nrag
import nifty.hdf5 as nh5
import vigra
from functools import partial


class TestAccumulation(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        urlData = 'https://www.dropbox.com/s/l1tgzlim8h1pb7w/test_data_anisotropic.zip?dl=0'
        call(['wget', '-Odata.zip', urlData])
        with zipfile.ZipFile('./data.zip') as f:
            f.extractall('.')
        os.remove('./data.zip')

        urlFeatures = 'https://www.dropbox.com/s/5nz0cjd9zdnfbmy/reference_features.zip?dl=0'
        call(['wget', '-Ofeatures.zip', urlFeatures])
        with zipfile.ZipFile('./features.zip') as f:
            f.extractall('./features')
        os.remove('./features.zip')

        self.nNodes = vigra.readHDF5('./data/seg.h5', 'data').max() + 1
        self.segFile = nh5.openFile('./data/seg.h5')
        self.segArray = nh5.hdf5Array('uint32', self.segFile, 'data')
        self.dataFile = nh5.openFile('./data/pmap.h5')
        self.dataArray = nh5.hdf5Array('float32', self.dataFile, 'data')

    @classmethod
    def tearDownClass(self):
        nh5.closeFile(self.segFile)
        nh5.closeFile(self.dataFile)
        rmtree('./data')
        rmtree('./features')

    def makeToyData(self):
        shape = (2, 10, 10)
        seg = numpy.zeros(shape, dtype='uint32')
        val = numpy.zeros(shape, dtype='float32')

        # the test segmentation
        seg[0, :5] = 0
        seg[0, 5:] = 1
        seg[1, :5] = 2
        seg[1, 5:] = 3

        # the test values
        val[0, :] = 0.
        val[1, :] = 1.
        return seg, val

    # check for the xy - edges is the same for all three accumulations
    def checkToyFeats(self, feats, zDir):
        self.assertEqual(feats[0, 0], 0.)
        self.assertEqual(feats[3, 0], 1.)
        if zDir == 0:
            self.assertEqual(feats[1, 0], .5)
            self.assertEqual(feats[2, 0], .5)
        elif zDir == 1:
            self.assertEqual(feats[1, 0], 0.)
            self.assertEqual(feats[2, 0], 0.)
        elif zDir == 2:
            self.assertEqual(feats[1, 0], 1.)
            self.assertEqual(feats[2, 0], 1.)

    def testFlatAccumulation(self):
        seg, val = self.makeToyData()
        rag = nrag.gridRag(seg, numberOfLabels=seg.max() + 1)

        # test the different z accumulations
        for zDir in (0, 1, 2):
            feats = nrag.accumulateEdgeFeaturesFlat(rag, val, val.min(), val.max(), zDir, 1)
            self.assertEqual(len(feats), rag.numberOfEdges)
            self.checkToyFeats(feats, zDir)

    def testReferenceFeatures(self):
        rag = nrag.gridRagStacked2DHdf5(self.segArray, self.nNodes)

        def singleFunctionTest(feature_function, name):

            def singleFeatureTest(fu, typ, zDir):
                xname = 'feats_%s_%s_%i_xy.h5' % (name, typ, zDir)
                zname = 'feats_%s_%s_%i_z.h5' % (name, typ, zDir)
                xy_file = nh5.createFile(xname)
                z_file = nh5.createFile(zname)
                xy_shape = [
                    rag.totalNumberOfInSliceEdges if typ in ('xy', 'both') else 1,
                    9 if name == 'standard' else 9 * 12
                ]
                xy_chunks = [min(2500, xy_shape[0]), xy_shape[1]]
                z_shape = [
                    rag.totalNumberOfInBetweenSliceEdges if typ in ('z', 'both') else 1,
                    9 if name == 'standard' else 9 * 12
                ]
                z_chunks = [min(2500, z_shape[0]), z_shape[1]]
                xy_array = nh5.hdf5Array('float32', xy_file, 'data', xy_shape, xy_chunks)
                z_array = nh5.hdf5Array('float32', z_file, 'data', z_shape, z_chunks)
                fu(rag, self.dataArray, xy_array, z_array, zDirection=zDir)
                xfeats = xy_array.readSubarray([0, 0], xy_shape)
                zfeats = z_array.readSubarray([0, 0], z_shape)
                nh5.closeFile(xy_file)
                nh5.closeFile(z_file)
                os.remove(xname)
                os.remove(zname)
                return xname, zname, xfeats, zfeats

            for typ in ('both', 'xy', 'z'):
                if typ == 'both':
                    new_fu = partial(feature_function, keepXYOnly=False, keepZOnly=False)
                elif typ == 'xy':
                    new_fu = partial(feature_function, keepXYOnly=True, keepZOnly=False)
                elif typ == 'z':
                    new_fu = partial(feature_function, keepXYOnly=False, keepZOnly=True)

                if typ == 'z':
                    for zDir in (0, 1, 2):
                        _, zname, _, zfeats = singleFeatureTest(new_fu, typ, zDir)
                        ref_feats = vigra.readHDF5(os.path.join('./features', zname), 'data')
                        self.assertTrue(numpy.allclose(zfeats, ref_feats))

                else:
                    zDir = 0
                    xname, zname, xfeats, zfeats = singleFeatureTest(new_fu, typ, zDir)
                    ref_feats_xy = vigra.readHDF5(os.path.join('./features', xname), 'data')
                    self.assertTrue(numpy.allclose(xfeats, ref_feats_xy))
                    if typ == 'both':
                        ref_feats_z = vigra.readHDF5(os.path.join('./features', zname), 'data')
                        self.assertTrue(numpy.allclose(zfeats, ref_feats_z))

        # standard feats
        singleFunctionTest(nrag.accumulateEdgeStandardFeatures, 'standard')
        # filter feats
        singleFunctionTest(nrag.accumulateEdgeFeaturesFromFilters, 'filter')

    def makeReferenceFeaturesStackedRag(self):
        rag = nrag.gridRagStacked2DHdf5(self.segArray, self.nNodes)

        def makeFeatures(feature_function, name):

            def singleFeature(fu, typ, zDir):
                xname = './feats_%s_%s_%i_xy.h5' % (name, typ, zDir)
                zname = './feats_%s_%s_%i_z.h5' % (name, typ, zDir)
                xy_file = nh5.createFile(xname)
                z_file = nh5.createFile(zname)
                xy_shape = [
                    rag.totalNumberOfInSliceEdges if typ in ('xy', 'both') else 1,
                    9 if name == 'standard' else 9 * 12
                ]
                xy_chunks = [min(2500, xy_shape[0]), xy_shape[1]]
                z_shape = [
                    rag.totalNumberOfInBetweenSliceEdges if typ in ('z', 'both') else 1,
                    9 if name == 'standard' else 9 * 12
                ]
                z_chunks = [min(2500, z_shape[0]), z_shape[1]]
                xy_array = nh5.hdf5Array('float32', xy_file, 'data', xy_shape, xy_chunks)
                z_array = nh5.hdf5Array('float32', z_file, 'data', z_shape, z_chunks)
                fu(rag, self.dataArray, xy_array, z_array, zDirection=zDir)
                nh5.closeFile(xy_file)
                nh5.closeFile(z_file)
                return xname, zname

            for typ in ('both', 'xy', 'z'):
                if typ == 'both':
                    new_fu = partial(feature_function, keepXYOnly=False, keepZOnly=False)
                elif typ == 'xy':
                    new_fu = partial(feature_function, keepXYOnly=True, keepZOnly=False)
                elif typ == 'z':
                    new_fu = partial(feature_function, keepXYOnly=False, keepZOnly=True)

                if typ == 'z':
                    for zDir in (0, 1, 2):
                        xname, _ = singleFeature(new_fu, typ, zDir)
                        os.remove(xname)

                else:
                    zDir = 0
                    _, zname = singleFeature(new_fu, typ, zDir)
                    if typ == 'xy':
                        os.remove(zname)

        # reference standard feats
        makeFeatures(nrag.accumulateEdgeStandardFeatures, 'standard')

        # reference filter feats
        makeFeatures(nrag.accumulateEdgeFeaturesFromFilters, 'filter')

    @unittest.expectedFailure
    def testStandardAccumulation(self):
        seg, val = self.makeToyData()
        rag = nrag.gridRagStacked2D(seg)

        # test the different z accumulations
        for zDir in (0, 1, 2):
            feats = nrag.accumulateEdgeStandardFeatures(
                rag,
                val,
                zDir
            )
            self.assertEqual(len(feats), rag.numberOfEdges)
            self.checkToyFeats(feats, zDir)


if __name__ == '__main__':
    unittest.main()
