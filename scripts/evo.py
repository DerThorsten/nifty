from __future__ import  division,print_function
import numpy
import vigra
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def energy_surface_2d(shape , f=10):
    s_shape = [s // f for s in shape]
    r = numpy.random.rand(*s_shape).astype('float32')
    r = vigra.sampling.resize(r, shape)
    return r

# class SimpleModel(object):
#     def __init__(self,shape=[100,100]):
#         self.e = energy_surface_2d(shape=shape)
#         self.n_weights = 2
#     def __call__(self, weights):
#         x,y = weights[0],weights[1]
#         x = int(round(x,0))
#         y = int(round(y,0))
#         return self.e[x,y]
    

class SimpleModel(object):
    def __init__(self):
        self.n_weights = 2
    def __call__(self, weights):
        x,y = weights[0],weights[1]
        return x**2 + y**2 +2*x +8*y
class Optimizer(object):

    def __init__(self, model, sigma, lr=0.01):
        self.lr = lr
        self.model = model
        self.sigma = sigma
        self.n_weights = model.n_weights
        self.w = numpy.array([0.0,0.0])

    

    def optimize(self, steps=200000, callback=None):

        for step in range(steps):
            self.do_step(step)
            print('E: %f'%self.model(self.w),self.w)

    def do_step(self,step):

        gradient = self.estimate_gradient(step)
        elr = self.lr
        self.w =  self.w - ( elr * gradient)
        #self.w = numpy.clip(self.w,0,99)


    def estimate_gradient(self, step,samples_per_step=10):

        gradient = numpy.zeros(self.n_weights)
        c = 0 
        for s in range(samples_per_step):

            esigma = self.sigma
            # perturb weight
            eps  = numpy.random.normal(0, esigma,size=self.n_weights)
            w_p = self.w + eps


            # HACK
            #w_p = numpy.clip(w_p,0,99)

            # compute energy
            e = self.model(w_p)

            if not math.isnan(e):

                gradient += e*eps
                c += 1
        assert c != 0
        gradient /= (float(c)*esigma)

        return gradient


if __name__ == '__main__':

    numpy.random.seed(11)
    model  = SimpleModel()






    optimizer = Optimizer(model=model,sigma=0.1)
    optimizer.optimize()




    print(model(numpy.array([1,1])))