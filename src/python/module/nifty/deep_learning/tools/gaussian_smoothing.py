# import math
# import numpy
# from scipy.stats import norm
# import keras
# import keras.backend as K

# def make_kernel(sigma, r=None):
#     if r is None:
#         r =  3.5*float(sigma) + 0.5
#         r = int(r+0.5)


#     #print r

#     kernel = numpy.zeros(2*r+1)


#     for i in range(kernel.size):
#         d = abs(i-r)

#         kernel[i] = norm.pdf(d)

#     return kernel/kernel.sum()



# def gaussian_smoothing_2d(tensor_in, sigma, n_channels=None, r=None,  data_format=None):
#     assert keras.backend.backend() == 'tensorflow'

#     if n_channels is None:
#         n_channels = K.int_shape(tensor_in)[3]

#     # tensorflow excepts
#     # [filter_height, filter_width, in_channels, channel_multiplier]
#     # where channel_multiplier is 1 in our case

#     kernel = make_kernel(sigma=sigma, r=r)
#     l = len(kernel)
#     k0 = kernel[:, None,None,None]
#     k1 = kernel[None, :,None,None]

#     # repeat channel times
#     k0 = K.constant(numpy.concatenate([k0]*n_channels,axis=2))
#     k1 = K.constant(numpy.concatenate([k1]*n_channels,axis=2))

#     x = tensor_in#K._preprocess_conv2d_input(tensor_in)
#     x = K.tf.nn.depthwise_conv2d(x, k0, padding='SAME', strides=[1,1,1,1])
#     x = K.tf.nn.depthwise_conv2d(x, k1, padding='SAME', strides=[1,1,1,1])

#     return x
#     #return K._postprocess_conv2d_output(x, data_format)

# print make_kernel(0.75)