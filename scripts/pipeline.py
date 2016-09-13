import segmentation_pipeline as segp

trainInput = {
    'rawData' : 'train-volume.tif',
    'superpixels' : None,   # ('file.h5','data')
    'gt' : 'train-labels.tif',
}

testInput = {
    'rawData' : 'test-volume.tif',
    'superpixels' : None
}


settings = {
    'rootOutDir' : '/home/tbeier/src/nifty/scripts/out',

    ##########################################################
    # in case supervoxels are not providedragDir
    ##########################################################
    'spSigmaHessian' : 2.0,     # sigma for hessian of gaussian eigenvalues

    'reduceBy': 10,             # reduce #superpixel by this factor 
                                # via agglomerative clustering

    'sizeRegularizer' : 0.5,    # sizeRegularizer term to overseg, ragFile, settings
                                # make apporx. equal size supervoxels



    ##########################################################
    # edge gt threshold
    ##########################################################
    'fuztGtThreshold': (0.2, 0.8),

    ##########################################################
    #  debug settings
    ##########################################################
    'debug' : True
}




segp.runPipeline(trainInput=trainInput, testInput=testInput, settings=settings)

