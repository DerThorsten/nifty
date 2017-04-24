
_has_requirements = True
try:
    import tensorflow
    import pymalis
    import tensorflow 
    import tensorflow as tf
    from tensorflow.python.framework import ops

except ImportError:
    _has_requirements = False
    pass 
    


import numpy as np

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):


    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MalisLossGradient for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Def custom square function using np.square instead of tf.square:
def malis_loss(affinities, groundtruth, name=None):

    if not _has_requirements:
        raise RuntimeError("requirements not fulfilled: needs tensorflow and pymalis")
    def impl(a,gt):
        #grad_p, grad_n = pymalis.malis(gt, a)

        return 1.0#, #_deep_learning.malisLossAndGradient(a,gt,0.5)


    with ops.name_scope(name, "malis_loss", [affinities,groundtruth]) as name:
        ret_list = py_func(impl,
                        [affinities,groundtruth],
                        [tf.float32],
                        name=name,
                        grad=_MalisLossGradient)  # <-- here's the call to the gradient
        return ret_list[0]


# Def custom square function using np.square instead of tf.square:
def malis_loss_gradient(affinities, groundtruth, name=None):
    
    def impl(a,gt):
        grad_p, grad_n = pymalis.malis(a, gt)
        return grad_p + grad_n



    with ops.name_scope(name, "malis_loss", [affinities,groundtruth]) as name:
        ret_list = py_func(impl,
                        [affinities,groundtruth],
                        [tf.float32],
                        name=name,
                        grad=_MalisLossGradient)  # <-- here's the call to the gradient
        return ret_list[0]




# Actual gradient:
def _MalisLossGradient(op, grad_l):
    affinities = op.inputs[0]
    gt = op.inputs[1]
   
    return malis_loss_gradient(affinities, gt),None



# with tf.Session() as sess:
#     x = tf.constant([1., 2.])
#     y = mysquare(x)
#     tf.global_variables_initializer().run()

#     print(x.eval(), y.eval(), tf.gradients(y, x)[0].eval())



