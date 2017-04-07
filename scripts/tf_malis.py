from  __future__ import print_function,division

import nifty
import nifty.deep_learning
import tensorflow as tf

import numpy

with tf.Session() as sess:

    #2D 2x1
    gt = [
        [0,1],
        [0,1],
    ]

    aff = [
        [[0,0],[0,0]],
        [[0,0],[0,0]]
    ]
    aff = numpy.array(aff,dtype='float32')


    aff = tf.constant(aff)
    gt = tf.constant(gt)
    loss= nifty.deep_learning.malis_loss(aff,gt)

    tf.global_variables_initializer().run()
    print(aff.eval(), gt.eval(), tf.gradients(loss, aff)[0].eval())
