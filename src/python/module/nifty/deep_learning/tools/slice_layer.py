
try:
    import keras
    from keras import backend as K
    from keras.layers import *
    from keras.layers.core import *
except ImportError:
    pass

def slice_features_2d(x, f_begin, f_end , shape_2d=None):
    
    if shape_2d is None:
        s = K.int_shape(x)
        output_shape = [int(s[1]),int(s[2])]
        slicing =[
            slice(None),
            slice(0,int(s[1]) ),
            slice(0,int(s[2]) ),
            slice(f_begin, f_end)
        ]
    else:
        output_shape = list(shape_2d)
        slicing =[
            slice(None),
            slice(0,int(shape_2d[0]) ),
            slice(0,int(shape_2d[1]) ),
            slice(f_begin, f_end)
        ]
    def slice_function(xx):
        return xx[slicing]

    return Lambda(function=slice_function, output_shape=output_shape+[f_end-f_begin])(x)
    #return SliceLayer(slicing)(x)