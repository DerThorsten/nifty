import segmentation_pipeline as segp

trainInput = {
    'rawData' : 'train-volume.tif',
    'superpixels' : ('wsdt_interceptor_train.h5','data'),
    'pmap' : ('interceptor_train.h5','data'),
    'gt' : 'train-labels.tif',
}

testInput = {
    'rawData' : 'test-volume.tif',
    'superpixels' : ('wsdt_interceptor_test.h5','data'),
    'pmap' : ('interceptor_test.h5','data'),
}


settings = {
    'rootOutDir' : '/home/tbeier/src/nifty/scripts/out',

    ##########################################################
    # in case supervoxels are not providedragDir
    ##########################################################
    'spSigmaHessian' : 3.0,     # sigma for hessian of gaussian eigenvalues

    'reduceBy': 15,             # reduce #superpixel by this factor 
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
    'debug' : True,


    ##########################################################
    #  classifier local
    ##########################################################
    'clfLocal' :{
        'nRounds' : 500,
        'maxDepth' : 3,
        'getApproxError' : True
    },

    ##########################################################
    #  classifier lifterd
    ##########################################################
    'clfLifted' :{
        'nRounds' : 500,
        'maxDepth' : 3,
        'getApproxError' : True
    },






    ##########################################################
    #  lifted multicut solver / objective
    ##########################################################
    'betaLocal'  : 0.5,
    'betaLifted' : 0.5,
    'gamma' : 0.5, # higher gamma gives local weights more power
}




segp.runPipeline(trainInput=trainInput, testInput=testInput, settings=settings)

