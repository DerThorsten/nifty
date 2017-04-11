import keras
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np






class Dilation2D(Layer):

    def __init__(self, filters, kernel_size, padding='same', strides=None, name=None, dilation_rate=None, **kwargs):
        super(Dilation2D, self).__init__(**kwargs)

        assert padding == 'same'
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides        
        if self.strides is None:
            self.strides = (1,1)

        self.dilation_rate = dilation_rate
        if self.dilation_rate is None:
            self.dilation_rate = (1,1)

    def build(self, input_shape):


        # Create a trainable weight variable for this layer.
        k_shape = (self.kernel_size[0], self.kernel_size[1], self.filters)
        self.kernel = self.add_weight(shape=k_shape,
                                      initializer='uniform',
                                      trainable=True)
        super(Dilation2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        rates = [1,self.dilation_rate[0], self.dilation_rate[1],1]
        strides = [1,self.strides[0],self.strides[1],1]
        padding = 'VALID'
        if self.padding == 'same':
            padding = 'SAME'
        elif self.padding == 'valid':
            padding = 'VALID'

        return K.tf.nn.dilation2d(x, self.kernel, strides=strides, rates=rates, 
            padding=padding, name=self.name)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)



class Conv1x1Dilation2D(Layer):
    def __init__(self, filters, kernel_size, padding='same', strides=None, name=None, dilation_rate=None,**kwargs):
        super(Conv1x1Dilation2D, self).__init__(**kwargs)
        self.filters = filters
        self.dilation_layer = Dilation2D(filters=filters, kernel_size=kernel_size,
            padding=padding, strides=strides, name=name, dilation_rate=dilation_rate)

    def call(self, x):
        x = keras.layers.convolutional.Conv2D(filters=self.filters, kernel_size=[1,1], padding='valid')(x)
        x =  self.dilation_layer(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)
