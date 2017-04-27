import numpy
import vigra
import nifty.graph.rag as nrag

def test_accumulation_flat():
    shape = (2,10,10)
    seg = numpy.zeros( shape, dtype = 'uint32')
    val = numpy.zeros( shape, dtype = 'float32')

    # the test segmentation
    seg[0,:5] = 0
    seg[0,5:] = 1
    seg[1,:5] = 2
    seg[1,5:] = 3
    rag = nrag.gridRag(seg)

    print rag.uvIds()

    # the test values
    val[0,:] = 0.
    val[1,:] = 1.

    # test the different z accumulations
    for zDir in (0,1,2):
        feats = nrag.accumulateEdgeFeaturesFlat(rag, val, val.min(), val.max(), zDir, 1)
        # check for the xy - edges is the same for all three accumulations
        assert feats[0,0] == 0.
        assert feats[3,0] == 1., str(feats[3,0])
        if zDir == 0:
            assert feats[1,0] == .5
            assert feats[2,0] == .5
        elif zDir == 1:
            assert feats[1,0] == 0.
            assert feats[2,0] == 0.
        elif zDir == 2:
            assert feats[1,0] == 1.
            assert feats[2,0] == 1.


if __name__ == '__main__':
    test_accumulation_flat()
