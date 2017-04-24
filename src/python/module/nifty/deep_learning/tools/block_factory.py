
try:
    import keras
    import keras.layers.convolutional as k_conv
    from keras import backend as K
    from slice_layer import  slice_features_2d
    from .. keras_extensions import *
except ImportError:
    pass
    
class BlockFactory(object):

    def __init__(self, **kwargs):

        self.activation_type = kwargs.get('activation_type','prelu')
        self.shared_axes = kwargs.get('shared_axes',[1,2])
        self.batch_norm_pre = False
        self.batch_norm_post = False

        base = self

    def activation(self, **kwargs):

        at = self.activation_type
        at = kwargs.get('activation', at)
        sa = kwargs.pop('shared_axes',[1,2])
        alpha = kwargs.pop('alpha',0.1)

        if at in ['relu','sigmoid','softmax']:
            return keras.layers.core.Activation(at)
        if at == 'none':
            class Op(object):
                def __call__(self, inputs):
                    return inputs
            return  Op()
        elif at == 'prelu':
            return keras.layers.advanced_activations.PReLU(shared_axes=sa)
        elif at == 'elu':
            return keras.layers.advanced_activations.ELU(alpha=alpha)
        elif at == 'cprelu':

            class Op(object):
                def __call__(self, inputs):

                    neg_inputs  = keras.layers.Lambda(lambda x: x * -1.0)(inputs)
                    if sa == [1,2]:
                        inputs = keras.layers.Concatenate(axis=3)([inputs, neg_inputs])
                    elif sa == [1,2,3]:
                        inputs = keras.layers.Concatenate(axis=4)([inputs, neg_inputs])

                    #return keras.layers.core.Activation('relu')(inputs)
                    return keras.layers.advanced_activations.PReLU(shared_axes=sa)(inputs)

                    #return inputs
            return  Op()

                

        elif at == 'prelu-softmax':
            class Op(object):
                def __call__(self, inputs):
                    inputs = keras.layers.advanced_activations.PReLU(shared_axes=sa)(inputs)
                    inputs = keras.layers.core.Activation('softmax')(inputs)
                    return inputs
            return  Op()
        else:
            raise RuntimeError('"%s" is an invalid activation type'%at )
   

    def shield2d(self, **kwargs):
        parent = self
        filters = kwargs.get('filters')
        activation_type = kwargs.get('activation','prelu')# self.activation_type)
        sa = kwargs.pop('shared_axes',[1,2])
        class Op(object):
            def __call__(self, inputs):
                inputs = k_conv.Conv2D(filters=filters,padding='same', kernel_size=[1,1])(inputs)
                return   parent.activation(activation=activation_type, shared_axes=sa)(inputs) 

        return Op()

    def convNd(self,ndim, **kwargs):

        parent = self

        class Op(object):
            def __init__(self, **kwargs):
                self.shield          = kwargs.pop('shield', False)
                self.batch_norm_pre  = kwargs.pop('batch_norm_pre', parent.batch_norm_pre)
                self.batch_norm_post = kwargs.pop('batch_norm_post', parent.batch_norm_pre)
                self.filters         = kwargs.get('filters')
                self.filters_shield  = kwargs.pop('filters_shield', self.filters)
                #if ndim == 2 and kwargs.get('kernel_size') == [1,1]:
                #    self.activation_type = kwargs.pop('activation', 'prelu')
                #else:
                self.activation_type = kwargs.pop('activation', parent.activation_type)
                self.padding         = kwargs.pop('padding', 'same')

           
                self.sa = kwargs.pop('shared_axes',range(1,ndim+1))

                self.kwargs = kwargs
            def __call__(self, inputs):
                if self.shield:
                    inputs = parent.shield2d(filters=self.filters_shield,shared_axes=self.sa)(inputs)

                use_bias = self.kwargs.pop('use_bias', True) and not self.batch_norm_pre

                if ndim == 1:
                    Conv = k_conv.Conv1D
                elif ndim == 2:
                    Conv = k_conv.Conv2D
                elif ndim == 3:
                    Conv = k_conv.Conv3D

                #print("the kwargs",self.kwargs,"dim",ndim,Conv)
                inputs = Conv(padding=self.padding, use_bias=use_bias, **self.kwargs)(inputs)
                if self.batch_norm_pre:
                    inputs = keras.layers.normalization.BatchNormalization(axis=ndim+1, center=True, scale=False)(inputs)
                inputs =  parent.activation(activation=self.activation_type, shared_axes=self.sa)(inputs)
                if self.batch_norm_post:
                    inputs = keras.layers.normalization.BatchNormalization(axis=ndim+1, center=True, scale=False)(inputs)
                return inputs

        return Op(**kwargs)

    def conv1d(self, **kwargs):
        return  self.convNd(1, **kwargs)

    def conv2d(self, **kwargs):
        return  self.convNd(2, **kwargs)

    def conv3d(self, **kwargs):
        return  self.convNd(3, **kwargs)


    




    def erosion2d(self, **kwargs):
        return self._dilation_erosion2d(mode='erosion', **kwargs)
    def dilation2d(self, **kwargs):
        return self._dilation_erosion2d(mode='erosion', **kwargs)



    def _dilation_erosion2d(self, **kwargs):
        parent = self
        class Op(object):
            def __init__(self, mode,  **kwargs):
                self.shield              = kwargs.pop('shield', True)
                self.nl_shield = kwargs.pop('nl_shield', True)
                self.filters = kwargs.get('filters')
                self.mode = mode
                self.kwargs = kwargs

            def __call__(self, inputs):
                x = inputs
                if self.shield :
                    if not self.nl_shield:
                        activation = 'none'
                    x = parent.conv2d(filters=self.filters, kernel_size=[1,1], padding='same')(x)
                return Erosion2D(**self.kwargs)(x)


        return Op(**kwargs)

    def residual_merge_2d(self, **kwargs):

        parent = self
        

        class Op(object):
            def __init__(self, mode='add'):
                self.mode = mode

            def __call__(self, inputs):
                assert len(inputs) == 2
                a, b = inputs
                print(a,b)
                if self.mode != 'add':
                    nf_a = K.int_shape(a)[3]
                    nf_b = K.int_shape(b)[3]

                    if self.mode == 'max_add':
                        if(nf_a < nf_b):
                            a = parent.conv2d(filters=nf_b,kernel_size=[1,1])(a)
                        elif(nf_b < nf_a): 
                            b = parent.conv2d(filters=nf_a,kernel_size=[1,1])(b)
                    elif self.mode == 'min_add':
                        if(nf_a > nf_b):
                            a = parent.conv2d(filters=nf_b,kernel_size=[1,1])(a)
                        elif(nf_b > nf_a): 
                            b = parent.conv2d(filters=nf_a,kernel_size=[1,1])(b)
                return keras.layers.Add()([a,b])
                    

        return Op(**kwargs)



    def pseudo_separable_conv2d(self, filters, kernel_size, **kwargs):

        parent = self
        

        class Op(object):
            def __init__(self, filters, kernel_size, **kwargs):
                self.filters = filters
                self.kernel_size = kernel_size
                self.kwargs = kwargs

            def __call__(self, inputs):

                dr = self.kwargs.pop('dilation_rate',1)
                dr_a = [dr, 1]
                dr_b = [1, dr]

                a = parent.conv2d(filters=self.filters//2, kernel_size=[self.kernel_size, 1], dilation_rate=dr_a, **self.kwargs)(inputs)
                a = parent.conv2d(filters=self.filters//2, kernel_size=[1, self.kernel_size], dilation_rate=dr_b, **self.kwargs)(a)
                b = parent.conv2d(filters=self.filters//2, kernel_size=[self.kernel_size, 1], dilation_rate=dr_a, **self.kwargs)(inputs)
                b = parent.conv2d(filters=self.filters//2, kernel_size=[1, self.kernel_size], dilation_rate=dr_b, **self.kwargs)(b)

                return keras.layers.Concatenate(axis=3)([a,b])

        return Op(filters=filters, kernel_size=kernel_size, **kwargs)

  

    def conv2d_transpose(self, **kwargs):

        parent = self

        class Op(object):
            def __init__(self, **kwargs):
                self.shield          = kwargs.pop('shield', False)
                self.batch_norm_pre  = kwargs.pop('batch_norm_pre', False)
                self.batch_norm_post = kwargs.pop('batch_norm_post', False)
                self.filters         = kwargs.get('filters')
                self.filters_shield  = kwargs.pop('filters_shield', self.filters)
                self.activation_type = kwargs.pop('activation', parent.activation_type)
                self.padding         = kwargs.pop('padding', 'same')

           
                self.sa = kwargs.pop('shared_axes', [1,2])

                self.kwargs = kwargs
            def __call__(self, inputs):
                if self.shield:
                    inputs = parent.shield2d(filters=self.filters_shield,shared_axes=self.sa)(inputs)

                use_bias = self.kwargs.pop('use_bias', True) and not self.batch_norm_pre

    
                Conv = k_conv.Conv2D
                #print("the kwargs",self.kwargs,"dim",ndim,Conv)
                inputs = k_conv.Conv2DTranspose(padding=self.padding, use_bias=use_bias, **self.kwargs)(inputs)
                if self.batch_norm_pre:
                    inputs = keras.layers.normalization.BatchNormalization(axis=3, center=True, scale=False)(inputs)
                inputs =  parent.activation(activation=self.activation_type, shared_axes=self.sa)(inputs)
                if self.batch_norm_post:
                    inputs = keras.layers.normalization.BatchNormalization(axis=3, center=True, scale=False)(inputs)
                return inputs

        return Op(**kwargs)




    def giant_inceptor_block(self, **kwargs):

        parent = self
        residual =  kwargs.get('residual', True)

        filters_shield = kwargs.get('filters_shield', 10)
        filters = kwargs.get('filters', 6*filters_shield)
         
        class Op(object):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, inputs):


                def dropout(x):
                    x =  keras.layers.core.Dropout( kwargs.get('p_dropout', 0.1))(x)
                    x =  keras.layers.core.SpatialDropout2D( kwargs.get('p_dropout_spatial', 0.25))(x)
                    return x


                if residual:
                    # shield
                    x_in = parent.conv2d(shield=False, filters=filters, kernel_size=[1,1])(inputs)


                # TOWERS
                towers = []


                # 1x1 simple
                x = parent.conv2d(shield=False, filters=filters_shield, kernel_size=[1,1])(dropout(inputs))
                towers.append(x)

            
                #  shielded 3x3 
                x = parent.conv2d(shield=True, filters=filters_shield, kernel_size=[3,3])(dropout(inputs))
                towers.append(x)

                #  shielded 5x5
                x = parent.conv2d(shield=True, filters=filters_shield, kernel_size=[5,5])(dropout(inputs)) 
                towers.append(x)
       
                #  shielded 3x3 dilated 
                if self.kwargs.get("use_dc_3x3", True):
                    x = parent.conv2d(shield=True, filters=filters_shield,
                        kernel_size=[3,3], dilation_rate=(2,2))(dropout(inputs))

                    x = parent.conv2d(shield=False, filters=filters_shield//2, 
                        kernel_size=[3,3])(x)

                    towers.append(x)


                #  shielded 5x5 dilated
                if self.kwargs.get("use_dc_5x5", True):
                    x = parent.conv2d(shield=True, filters=filters_shield//2, 
                        kernel_size=[5,5], dilation_rate=(2,2))(dropout(inputs))

                    x = parent.conv2d(shield=False, filters=filters_shield//2, 
                        kernel_size=[3,3])(x)

                    towers.append(x)


                if self.kwargs.get("use_7x1", True):
                    # 7x1 / 1x7 shield    
                    x = parent.pseudo_separable_conv2d(shield=True, filters=filters_shield//2, kernel_size=7)(dropout(inputs))
                    towers.append(x)


                if self.kwargs.get("use_dc_7x1", True):
                    # 7x1 / 1x7 shield    
                    x = parent.pseudo_separable_conv2d(shield=True, filters=filters_shield//2, kernel_size=7, dilation_rate=2)(dropout(inputs))
                    x = parent.conv2d(shield=False, filters=filters_shield//2, 
                        kernel_size=[3,3])(x)
                    towers.append(x)

                    
                # max-pool 3x3 and shield
                if self.kwargs.get("use_mp_3x3", True):
                    x = keras.layers.pooling.MaxPooling2D((3, 3), strides=(1, 1), 
                        padding='same')(dropout(inputs))   
                    x = parent.conv2d(shield=False, filters=filters_shield, 
                        kernel_size=[1,1])(dropout(inputs))  
                    towers.append(x)
                   
                # max-pool 5x5 and shield
                if self.kwargs.get("use_mp_5x5", True):
                    x = keras.layers.pooling.MaxPooling2D((5, 5), strides=(1, 1), padding='same')(dropout(inputs))   
                    x = parent.conv2d(shield=False, filters=filters_shield//2, kernel_size=[1,1])(dropout(inputs))  
                    towers.append(x)


                # concatenate towers
                x = keras.layers.Concatenate(axis=3)(towers)

                # shield output ?!?
                x = parent.conv2d(batch_norm_pre=True, batch_norm_post=True, shield=False, filters=filters, kernel_size=[1,1])(x)

                if residual:
                    # residual
                    x = keras.layers.Add()([x_in, x])

                return x

        return Op(**kwargs)



    def deep_inceptor_block(self, **kwargs):

        parent = self
        residual =  kwargs.get('residual', True)

        filters_shield = kwargs.get('filters_shield', 10)
        filters = kwargs.get('filters', 6*filters_shield)
         
        class Op(object):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, inputs):


                def dropout(x):
                    x =  keras.layers.core.Dropout( kwargs.get('p_dropout', 0.1))(x)
                    x =  keras.layers.core.SpatialDropout2D( kwargs.get('p_dropout_spatial', 0.15))(x)
                    return x


                if residual:
                    # shield
                    x_in = parent.conv2d(shield=False, filters=filters, kernel_size=[1,1])(inputs)


                # TOWERS
                towers = []




                
                ###################################
                ## VERY DEEP SUBBLOCK
                ## 
                ## => the main cost of this block 
                ##    should be then 5x5  => extends the field of view by 25 pixels
                ###################################
            
                # make very large  
                x_a = parent.pseudo_separable_conv2d(shield=True, filters=filters_shield//4,                 kernel_size=7, dilation_rate=1)(dropout(inputs))
                x_b = parent.pseudo_separable_conv2d(shield=True, filters=filters_shield- filters_shield//4, kernel_size=7, dilation_rate=2)(dropout(inputs))

                # make very fine no dilation what so ever
                x_c = parent.conv2d(shield=True, filters=filters_shield//2,kernel_size=[3,3])(dropout(inputs))
                x_c = parent.conv2d(shield=True, filters=filters_shield//2,kernel_size=[3,3])(x_b)

                # concatenate.. no we have fine are large 
                x = keras.layers.Concatenate(axis=3)([x_a, x_b, x_c])


                x = parent.conv2d(shield=False,  filters=  filters_shield, kernel_size=[5,5], dilation_rate=[2,2])(x) 
                x = parent.conv2d(shield=False,  filters=  filters_shield, kernel_size=[3,3],                    )(x) 
                x = parent.conv2d(shield=False,  filters=  filters_shield, kernel_size=[5,5], dilation_rate=[1,1])(x) 
                x = parent.conv2d(shield=False,  filters=2*filters_shield, kernel_size=[3,3], dilation_rate=[1,1])(x) 




                ###################################
                ## VERY WIDE
                ###################################

                # 1x1 simple
                x = parent.conv2d(shield=False, filters=filters_shield, kernel_size=[1,1])(dropout(inputs))
                towers.append(x)


                #  shielded 3x3 
                x = parent.conv2d(shield=True, filters=filters_shield, kernel_size=[3,3])(dropout(inputs))
                towers.append(x)

                #  shielded 5x5
                x = parent.conv2d(shield=True, filters=filters_shield//2, kernel_size=[5,5])(dropout(inputs)) 
                towers.append(x)
       
                #  shielded 3x3 dilated 
                if self.kwargs.get("use_dc_3x3", True):
                    x = parent.conv2d(shield=True, filters=filters_shield//4,
                        kernel_size=[3,3], dilation_rate=(2,2))(dropout(inputs))
                    x = parent.conv2d(shield=False, filters=filters_shield//2, 
                        kernel_size=[3,3])(x)
                    towers.append(x)


                #  shielded 5x5 dilated
                if self.kwargs.get("use_dc_5x5", True):
                    x = parent.conv2d(shield=True, filters=filters_shield//4, 
                        kernel_size=[5,5], dilation_rate=(2,2))(dropout(inputs))
                    x = parent.conv2d(shield=False, filters=filters_shield//2, 
                        kernel_size=[3,3])(x)
                    towers.append(x)


                    
                # max-pool 3x3 and shield
                if self.kwargs.get("use_mp_3x3", True):
                    x = keras.layers.pooling.MaxPooling2D((3, 3), strides=(1, 1), 
                        padding='same')(dropout(inputs))   
                    x = parent.conv2d(shield=False, filters=filters_shield, 
                        kernel_size=[1,1])(dropout(inputs))  
                    towers.append(x)
                   
                # max-pool 7x7 and shield
                if self.kwargs.get("use_mp_5x5", True):
                    x = keras.layers.pooling.MaxPooling2D((5, 5), strides=(1, 1), padding='same')(dropout(inputs))   
                    x = parent.conv2d(shield=False, filters=filters_shield//4, kernel_size=[1,1])(dropout(inputs))  
                    towers.append(x)


                # concatenate towers
                x = keras.layers.Concatenate(axis=3)(towers)

                # shield output 
                x = parent.conv2d(batch_norm_pre=True, batch_norm_post=True, shield=False, filters=filters, kernel_size=[1,1])(x)

                if residual:
                    # residual
                    x = keras.layers.Add()([x_in, x])

                return x

        return Op(**kwargs)
        

    def cremi_start_block(self, input_shape, **kwargs):
        parent = self

        class Op(object):
            def __init__(self, **kwargs):

                self.use_3d        = kwargs.get('use_3d', True)
                self.use_central_1 = kwargs.get('use_central_1', True)
                self.use_central_3 = kwargs.get('use_central_3', True)
                self.kwargs = kwargs

            def __call__(self, input_tensor):

                # central / middle slice
                central_z = (input_shape[2] - 1)//2
                n_channels = input_shape[2]

                towers = []


                def dropout(x):
                    #p = kwargs.get('p_dropout', 0.1)
                    #x =  keras.layers.core.Dropout(p)(x)
                    return x 



                # real 3D convolution 
                # usually 2 of them (if input has 5 channels)
                if self.use_3d:
                    n_channels = input_shape[2]
                    assert n_channels % 2 == 1
                    n_conv = (n_channels - 1) // 2
                    shape_3d = input_shape + (1,)

                    x = keras.layers.core.Reshape(shape_3d)(dropout(input_tensor))
                    x = k_conv.ZeroPadding3D((n_conv,n_conv,0))(x)
                    for i in range(2):
                        x = parent.conv3d(filters=16, kernel_size=[3,3,3], padding='valid')(x)
                        #print("the shape",x.get_shape())
                    x  = keras.layers.core.Reshape( (input_shape[0], input_shape[1], 16))(x)
                    towers.append(x)


                ### central 3 slices
                if self.use_central_3 and n_channels >= 5:
                    x = slice_features_2d(input_tensor, central_z-1, central_z+2)

                    x0 = parent.conv2d(shield=False, filters=32, kernel_size=[3,3]                     )(dropout(x))
                    x1 = parent.conv2d(shield=False, filters=16, kernel_size=[3,3], dilation_rate=[2,2])(dropout(x))


                    towers += [x0, x1]

                ### central 1 slice 
                if self.use_central_1 and n_channels >= 5:
                    x = slice_features_2d(input_tensor, central_z, central_z+1)

                    x0 = parent.conv2d(shield=False, filters=16, kernel_size=[3,3]                     )(dropout(x))
                    x1 = parent.conv2d(shield=False, filters=16, kernel_size=[3,3], dilation_rate=[2,2])(dropout(x))
                    x2 = parent.conv2d(shield=False, filters=8,  kernel_size=[5,5])                     (dropout(x))

                    towers += [x0, x1, x2]


                ### use all
                if True:
                    x = input_tensor
                    x0 = parent.conv2d(shield=False, filters=16, kernel_size=[3,3]                     )(dropout(x))
                    x1 = parent.conv2d(shield=False, filters=16, kernel_size=[3,3], dilation_rate=[2,2])(dropout(x))
                    towers += [x0, x1]



                # dilation erosion
                x = parent.erosion2d(shield=False,filters=n_channels, kernel_size=[5,5])(input_tensor)
                towers += [x]
                x = parent.dilation2d(shield=False,filters=n_channels, kernel_size=[5,5])(input_tensor)
                towers += [x]



                if len(towers)>1:
                    # concatenate towers
                    x = keras.layers.Concatenate(axis=3)(towers)
                else:
                    x = towers[0]
                return x

        return Op(**kwargs)


    def cheap_wide_and_deep_inceptor_block(self, **kwargs):

        parent = self
        residual =  kwargs.get('residual', True)

        filters_shield = kwargs.get('filters_shield', 16)
        filters = kwargs.get('filters', int(5.5*float(filters_shield)+0.5) )

         
        class Op(object):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, inputs):


                #print("filters_shield",filters_shield,"fiters",filters)

                def dropout(x):
                    x =  keras.layers.core.Dropout( kwargs.get('p_dropout', 0.1))(x)
                    x =  keras.layers.core.SpatialDropout2D( kwargs.get('p_dropout_spatial', 0.15))(x)
                    return x


                if residual:
                    # shield
                    x_in = parent.conv2d(shield=False, filters=filters, kernel_size=[1,1])(inputs)


                # TOWERS
                towers = []




                ###################################
                ##  DEEP SUBBLOCK
                ###################################
            
                # make very large  
                x_a = parent.pseudo_separable_conv2d(shield=True, filters=filters_shield//4, kernel_size=7, dilation_rate=1)(dropout(inputs))
                x_b = parent.pseudo_separable_conv2d(shield=True, filters=filters_shield//4, kernel_size=7, dilation_rate=1)(dropout(inputs))

                # make very fine no dilation what so ever
                x_c = parent.conv2d(shield=True, filters=filters_shield//2,kernel_size=[3,3])(dropout(inputs))
                x_c = parent.conv2d(shield=True, filters=filters_shield//2,kernel_size=[3,3])(x_c)

                # concatenate.. no we have fine are large 
                x = keras.layers.Concatenate(axis=3)([x_a, x_b])


                x = parent.conv2d(shield=False,  filters=filters_shield, kernel_size=[3,3], dilation_rate=[2,2])(x) 
                x = parent.conv2d(shield=False,  filters=filters_shield, kernel_size=[3,3],                    )(x) 


                towers.append(x)


                ###################################
                ## WIDE
                ###################################

                # 1x1 simple
                x = parent.conv2d(shield=False, filters=filters_shield, kernel_size=[1,1])(dropout(inputs))
                towers.append(x)

                #  shielded 3x3 
                x = parent.conv2d(shield=True, filters=filters_shield, kernel_size=[3,3])(dropout(inputs))
                towers.append(x)

       
                #  shielded 3x3 dilated 
                if self.kwargs.get("use_dc_3x3", True):
                    x = parent.conv2d(shield=True, filters=filters_shield,
                        kernel_size=[3,3], dilation_rate=(2,2))(dropout(inputs))
                    x = parent.conv2d(shield=False, filters=filters_shield, 
                        kernel_size=[3,3])(x)
                    towers.append(x)


                # max-pool 3x3 and shield
                if self.kwargs.get("use_mp_3x3", True):
                    x = keras.layers.pooling.MaxPooling2D((3, 3), strides=(1, 1), 
                        padding='same')(dropout(inputs))   
                    x = parent.conv2d(shield=False, filters=filters_shield, 
                        kernel_size=[1,1])(dropout(inputs))  
                    towers.append(x)
                   
                # max-pool 7x7 and shield
                if self.kwargs.get("use_mp_7x7", True):
                    x = keras.layers.pooling.MaxPooling2D((7, 7), strides=(1, 1), padding='same')(dropout(inputs))   
                    x = parent.conv2d(shield=False, filters=filters_shield//2, kernel_size=[1,1])(dropout(inputs))  
                    towers.append(x)


                # concatenate towers
                x = keras.layers.Concatenate(axis=3)(towers)

                # shield output 
                x = parent.conv2d(batch_norm_pre=False, batch_norm_post=False, shield=False, filters=filters, kernel_size=[1,1])(x)

                if residual:
                    # residual
                    x = keras.layers.Add()([x_in, x])

                return x

        return Op(**kwargs)
        



    def high_res_inceptor_block(self, **kwargs):
        parent = self
        class Op(object):
            def __init__(self, **kwargs):

                self.filters_shield = kwargs.get('filters_shield', 16)
                #self.filters = kwargs.get('filters_shield', 2*self.filters_shield)
                self.n_conv_us = kwargs.get('n_conv_us', 3)
                self.n_conv = kwargs.get('n_conv', 3)
                self.kwargs = kwargs

            def __call__(self, inputs):

                fs = self.filters_shield


                def dropout(x):
                    #x =  keras.layers.core.Dropout( kwargs.get('p_dropout', 0.1))(x)
                    #x =  keras.layers.core.SpatialDropout2D( kwargs.get('p_dropout_spatial', 0.15))(x)
                    return x

                towers = []

                ##################################
                # up-sample ->  conv -> pool
                ##################################
                upsampled = parent.conv2d_transpose(shield=True, filters=fs, kernel_size=[2,2], strides=[2,2])(dropout(inputs))
                x = upsampled
                for i in range(self.n_conv_us):
                    x = parent.conv2d(filters=fs, kernel_size=[3,3])(x)
                x = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2))(x)
                towers.append(x)


                ##################################
                # conv
                ##################################
                x = inputs
                for i in range(self.n_conv):
                    x = parent.conv2d(shield=i==0,filters=fs, kernel_size=[3,3])(x)
                towers.append(x)

                x = keras.layers.Concatenate(axis=3)(towers)

                parent.residual_merge_2d(mode='max_add')([inputs, x])


                return x

        return Op(**kwargs)
