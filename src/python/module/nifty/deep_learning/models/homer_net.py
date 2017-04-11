from __future__ import division, print_function
import keras
from keras import backend as K
from ..tools.block_factory import BlockFactory

from  ..keras_extensions import *


class HomerNet(object):
    def __init__(self, loss_function, **kwargs):

        self.loss_function = loss_function
        self.kwargs = kwargs
        self.input_patch_shapes = self.kwargs['input_patch_shapes']
        self.n_resolutions = len(self.kwargs['input_patch_shapes'])
        self._bf = BlockFactory(activation_type='elu')
        self.model = self._make_net()

    def _make_net(self):

        input_patch_shapes = self.kwargs['input_patch_shapes']
        n_inputs  = len(input_patch_shapes)
        raw_inputs = [keras.layers.Input(input_patch_shapes[i]) for i in range(n_inputs)]



        ###########################################
        ###########################################
        ## FOV - LAYERS                          ##
        ###########################################
        ###########################################

        # start with the lowest resolution
        for ri in reversed(range(n_inputs)):

            # do the input layer
            input_conv = self.input_layer_conv(raw_inputs[ri], ri=ri)


            # some erosion
            #input_conv = self._bf.erosion2d(filters=10, kernel_size=[3,3])(input_conv)



            # the lowest resolution 
            if ri + 1 == n_inputs:
                fov = self.fov_layer(input_conv, ri=ri)
                fov = self.upsample_and_crop(fov, ri=ri)

            # mid res.
            elif ri !=0:
                # the input fov is the fov from the above scope
                fov = self.merge_fov_and_input_conv(fov, input_conv)
                fov = self.fov_layer(fov, ri=ri)
                bfov = self.upsample_and_crop(fov, ri=ri)

            #highest rest
            else:
                fov = self.merge_fov_and_input_conv(fov, input_conv)
    

        ###########################################
        ###########################################
        ## HR LAYERS                            ##
        ###########################################
        ###########################################

        # from here on we are done with the fv
        ht = self.hr_layer(fov)


        # make some output
        n_out = self.loss_function.expected_channels()
        a_out = self.loss_function.activation()
        out = keras.layers.convolutional.Convolution2D(filters=n_out, kernel_size=(1, 1),padding='same')(ht)
        out = a_out(out)



        model = keras.models.Model(inputs=raw_inputs, outputs=out)
        #opt = keras.optimizers.Adam()  
        #model.compile(optimizer=opt, loss=self.loss_functions[0])

        return model
 
    def merge_fov_and_input_conv(self, fov, input_conv):
        return self._bf.residual_merge_2d(mode='max_add')([fov,input_conv])

    def upsample_and_crop(self, input_layer, ri):
        assert ri >= 1

        shape =  K.int_shape(input_layer)
        nf  = shape[3]

        
        # upsample by a factor of two    
        us = self._bf.conv2d_transpose(filters=nf, kernel_size=(2,2), strides=(2,2))(input_layer)
        # how much cropping is needed
        has_size    = self.input_patch_shapes[ri  ][1]*2
        should_size = self.input_patch_shapes[ri-1][1]
        diff =  has_size - should_size
        c = diff // 2

        assert diff > 0 
        assert diff % 2 == 0
        cropping = ((c,c),(c,c))
        return keras.layers.convolutional.Cropping2D(cropping=cropping)(us)

    def input_layer_conv(self, input_layer, ri, **kwargs):
        #return input_layer
        input_shape = self.input_patch_shapes[ri]
        return self._bf.cremi_start_block(input_shape=input_shape,use_3d=False)(input_layer)

        


    def hr_layer(self, input_layer):
        x = input_layer
        for i in range(2):
            #x = self._bf.conv2d(filters=80, kernel_size=[3,3])(x)
            x = self._bf.giant_inceptor_block(filters=60,filters_shield=10)(x)
        x =  self._bf.residual_merge_2d(mode='max_add')([x,input_layer])

        return self._bf.high_res_inceptor_block()(x)


    def fov_layer(self, input_layer, ri, **kwargs):

        if ri + 1 != self.n_resolutions:
            return self.stay_on_res_layer(input_layer, **kwargs)
        else:
            return self.unet(input_layer, **kwargs)
        



    def stay_on_res_layer(self, input_layer, **kwargs):
        x = input_layer
        for i in range(2):
            #x = self._bf.conv2d(filters=80, kernel_size=[3,3])(x)
            x = self._bf.cheap_wide_and_deep_inceptor_block(filters=80,filters_shield=16)(x)
        return self._bf.residual_merge_2d(mode='max_add')([x,input_layer])


    def unet(self, input_layer, **kwargs):

        u_depth = kwargs.get('u_depth',3)

        if True:
            def conv_block(x_in):


                if K.int_shape(x_in)[3] != 80:
                    x_in = self._bf.conv2d(filters=80, kernel_size=[1,1])(x_in)
                x = x_in

                for i in range(2):
                    #x = self._bf.conv2d(filters=100, kernel_size=[3,3])(x_in)
                    x = self._bf.cheap_wide_and_deep_inceptor_block(filters=80,filters_shield=16)(x)
                return keras.layers.Add()([x_in,x])


            def pool(x):
                return keras.layers.MaxPooling2D(pool_size=[2,2], padding='same')(x)

            def up_and_merge(to_us, h):
                nf = K.int_shape(to_us)[3]
                us = self._bf.conv2d_transpose(filters=nf, kernel_size=(2,2), strides=(2,2))(to_us)
                return keras.layers.Add( )([us, h])


            if u_depth == 2:
                x0 = conv_block(input_layer)
                p0 = pool(x0)

                # lowest part in the net
                x1 = conv_block(p0)

                # up block
                m0  = up_and_merge(x1, x0)
                ux0 = conv_block(m0)

                return ux0

            elif u_depth == 3:

                # down
                x0 = conv_block(input_layer)
                p0 = pool(x0)

                x1 = conv_block(p0)
                p1 = pool(x1)

                # lowest part in the ne
                x2 = conv_block(p1)

                # up block
                m1  = up_and_merge(x2, x1)
                ux1 = conv_block(m1)

                m0  = up_and_merge(ux1, x0)
                ux0 = conv_block(m0)

                return ux0

