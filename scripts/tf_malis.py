from  __future__ import print_function,division

import nifty
import nifty.deep_learning.loss
import tensorflow as tf

import numpy
import vigra
import pymalis


with tf.Session() as sess:

    shape = [10,10,10]
    size = 1
    for s in shape :
        size*=s

    gt = numpy.random.randint(low=0, high=2,size=size).astype('uint32').reshape(shape)
    gt = numpy.array(vigra.analysis.labelVolume(gt)).astype('int64')

    aff = numpy.random.rand(*(shape+[3]))
    aff = numpy.array(aff,dtype='float32')


    #pymalis.malis(aff,gt)

    aff = tf.constant(aff)
    gt = tf.constant(gt)
    loss= nifty.deep_learning.loss.malis_loss(aff, gt)

    tf.global_variables_initializer().run()
    print(aff.eval(), gt.eval(), tf.gradients(loss, aff)[0].eval())
