from __future__ import division
from ..tools import slice_features_2d, gaussian_smoothing_2d

import keras
from keras import backend as K
from functools import partial





class PixelWiseLossBase(object):
    def __init__(self):
        pass

    def expected_channels(self):
        raise NotImplementedError("'expected_channels' needs to be implemented")

    def activation(self):
        raise NotImplementedError("'activation' needs to be implemented")



class PixelwiseWeightedLoss2D(object):
    
    def __init__(self, input_shape, loss_function, cropping=None):
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.cropping = cropping
        self.__name__ = "my_loss"
    
    def __call__(self, y_true_and_weights, y_pred):
        # possible cropping
        if self.cropping is not None:
            crop = keras.layers.convolutional.Cropping2D(cropping=self.cropping)
            y_true_and_weights = crop(y_true_and_weights)
            y_pred = crop(y_pred)

        # extract the weights
        y_true, y_weights = self.split(y_true_and_weights)

        # call the actual loss function
        per_pixel_loss = self.loss_function(y_true, y_pred)

        # weighting
        loss = per_pixel_loss *y_weights
        loss = K.mean(loss)

        return loss

    def split(self, x):
        n_channels = self.input_shape[2]
        x_a = slice_features_2d(x, 0, n_channels//2, shape_2d=self.input_shape[0:2])
        x_b = slice_features_2d(x, n_channels//2, n_channels, shape_2d=self.input_shape[0:2])
        return x_a, x_b




class GaussianLoss(object):
    def __init__(self, loss_function, sigma=1.0, beta=0.5, window_ratio=3.5):
        self.loss_function = loss_function
        self.sigma = sigma
        self.beta = beta
        self.window_ratio = window_ratio

    def __call__(self, y_true, y_pred):
        r = self.window_ratio*self.sigma + 0.5
        g = partial(gaussian_smoothing_2d,  sigma=self.sigma,r=int(r+0.5))

        a = self.loss_function(y_true=y_true,y_pred=y_pred)
        b = self.loss_function(y_true=g(y_true), y_pred=g(y_pred))

        return self.beta*a + (1.0-self.beta)*b


class PDiff(object):
    def __init__(self, p=2):
        self.p = p

    def __call__(self, y_true, y_pred):
        if self.p == 1:
            return K.abs(y_true - y_pred)
        elif self.p == 2:
            return K.square(y_true - y_pred)
        else:
            raise RuntimeError("p must be 1 or 2")



class SmoothedPDiff(PixelwiseWeightedLoss2D):
    def __init__(self,input_shape,p=2, sigma=0.25, beta=0.5, window_ratio=3.5, cropping=None,**kwargs):
        loss_function = PDiff(p=p)
        if sigma > 0.0001:
            loss_function = GaussianLoss(loss_function=loss_function, sigma=sigma,beta=beta, window_ratio=window_ratio)

        super(SmoothedPDiff, self).__init__(input_shape=input_shape,loss_function=loss_function, cropping=cropping,**kwargs)




class LiftedEdgeLoss(SmoothedPDiff):
    def __init__(self,input_shape, n_channels, **kwargs):
        self.n_channels = n_channels
        super(LiftedEdgeLoss, self).__init__(input_shape=input_shape,**kwargs)


    def expected_channels(self):
        return self.n_channels

    def activation(self):
        return keras.layers.core.Activation('sigmoid')