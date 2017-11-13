import numpy




def affinities_to_weights(affinities, offsets):
    ndim = affinities.ndim 
    assert affinities.shape[-1] == offsets.shape[0]

    if ndim == 4:
        assert offsets.shape[1] == 3
    elif ndim == 3:
        assert offsets.shape[1] == 2
    else:
        assert False

    weights = affinities.copy()
    weights[:]       =        -2.0*(affinities[:]-0.5)
    weights[...,0:3] =  1.0 -  affinities[...,0:3]

    # local z 
    weights[...,0]*=0.2

    #non local z
    weights[...,3:7] *= 0.01

    w = (numpy.sum(numpy.abs(offsets)**2,1)**0.5)


    weights *= w


    return weights

