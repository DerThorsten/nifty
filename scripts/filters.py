import fastfilters
import numpy

tolFactor = 200.0

class HessianOfGaussianEv(object):
    def __init__(self, raw, sigma):
        self.sigma = sigma
        f = fastfilters.hessianOfGaussianEigenvalues(raw, self.sigma)
        self.mi = [numpy.min(f[:,:,:,c]) for c in range(3)]
        self.ma = [numpy.max(f[:,:,:,c]) for c in range(3)]
        self.range = [(ma - mi)/tolFactor for ma,mi in zip(self.ma, self.mi)]

        for c in  range(3):
            self.mi[c] -= self.range[c]
            self.ma[c] += self.range[c]

    def __call__(self, raw):
        f = fastfilters.hessianOfGaussianEigenvalues(raw, self.sigma)
        for c in  range(3):
            
            fc = f[:,:,:,c]
            fc -= self.mi[c]
            fc /= (self.ma[c] - self.mi[c])

        return f
       

class GaussianGradientMagnitude(object):
    def __init__(self, raw, sigma):
        self.sigma = sigma
        f = fastfilters.gaussianGradientMagnitude(raw, self.sigma)
        self.mi = numpy.min(f)
        self.ma = numpy.max(f)
        self.range = (self.ma - self.mi)/tolFactor
        self.mi -= self.range
        self.ma += self.range

    def __call__(self, raw):
        f = fastfilters.gaussianGradientMagnitude(raw, self.sigma)

        f -= self.mi
        f /= (self.ma - self.mi)

        return f[:,:,:,None]


class LaplacianOfGaussian(object):
    def __init__(self, raw, sigma):
        self.sigma = sigma
        f = fastfilters.laplacianOfGaussian(raw, self.sigma)
        self.mi = numpy.min(f)
        self.ma = numpy.max(f)
        self.range = (self.ma - self.mi)/tolFactor
        self.mi -= self.range
        self.ma += self.range

    def __call__(self, raw):
        f = fastfilters.laplacianOfGaussian(raw, self.sigma)

        f -= self.mi
        f /= (self.ma - self.mi)

        return f[:,:,:,None]


class GaussianSmoothing(object):
    def __init__(self, raw, sigma):
        self.sigma = sigma
        f = fastfilters.gaussianSmoothing(raw, self.sigma)
        self.mi = numpy.min(f)
        self.ma = numpy.max(f)
        self.range = (self.ma - self.mi)/tolFactor
        self.mi -= self.range
        self.ma += self.range

    def __call__(self, raw):
        f = fastfilters.gaussianSmoothing(raw, self.sigma)

        f -= self.mi
        f /= (self.ma - self.mi)

        return f[:,:,:,None]
