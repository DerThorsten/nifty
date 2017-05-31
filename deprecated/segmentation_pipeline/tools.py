import random
import numpy
import os
import h5py
import warnings
import multiprocessing
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from sklearn.cross_validation import KFold
import xgboost as xgb
from superpixels import *

from colorama import init, Fore, Back, Style
init()



def generateSplits(n, nSplits, frac=0.25):
    res = []
    for x in range(nSplits):
        indices = numpy.arange(n)
        numpy.random.shuffle(indices)

        na =  int(float(n)*frac)
        a = indices[0:na]
        b = indices[na:n]

        res.append((a, b))

    return res









class ThreadPoolExecutorStackTraced(ThreadPoolExecutor):

    def submit(self, fn, *args, **kwargs):
        """Submits the wrapped function instead of `fn`"""

        return super(ThreadPoolExecutorStackTraced, self).submit(
            self._function_wrapper, fn, *args, **kwargs)

    def _function_wrapper(self, fn, *args, **kwargs):
        """Wraps `fn` in order to preserve the traceback of any kind of
        raised exception

        """
        try:
            return fn(*args, **kwargs)
        except Exception:
            raise sys.exc_info()[0](traceback.format_exc())  # Creates an
                                                             # exception of the
                                                             # same type with the
                                                             # traceback as
                                                             # message



def printWarning(txt):
    print(Fore.RED + Back.BLACK + txt)
    print(Style.RESET_ALL)

def ensureDir(f):
    if not os.path.exists(f):
        os.makedirs(f)


def hasH5File(f, key=None):
    if os.path.exists(f):
        return True
    else:
        return False


def hasFile(f):
    if os.path.exists(f):
        return True
    else:
        return False

def isH5Path(h5path):
    if isinstance(h5path, tuple):
        if len(h5path) == 2:
            return True
    else:
        return False


def threadExecutor(nThreads = multiprocessing.cpu_count()):
    return ThreadPoolExecutorStackTraced(max_workers=nThreads) 



def h5Read(f, d='data'):
    f5 = h5py.File(f,'r')
    array = f5[d][:]
    f5.close()
    return array




class Classifier(object):
    def __init__(self, nRounds=200, maxDepth=3, nThreads=multiprocessing.cpu_count(), silent=1):

        self.param = {
            'bst:max_depth':maxDepth, 
            'bst:eta':1, 
            'silent':silent, 
            'num_class':2,
            'objective':'multi:softprob' 
        }
        self.param['nthread'] = nThreads
        
        self.nRounds = nRounds 
        self.nThreads = nThreads

    def train(self, X, Y, getApproxError=False):
        dtrain = xgb.DMatrix(X, label=Y)
        self.bst = xgb.train(self.param, dtrain, self.nRounds)

        if getApproxError:

            e = 0.0
            c = 0.0

            kf = KFold(Y.shape[0], n_folds=4)
            for train_index, test_index in kf:

                XTrain = X[train_index, :]
                XTest  = X[test_index, :]

                YTrain = Y[train_index]
                YTest  = Y[test_index]

                dtrain2 = xgb.DMatrix(XTrain, label=YTrain)
                bst = xgb.train(self.param, dtrain2, self.nRounds)
              

                dtest = xgb.DMatrix(XTest)
                probs = bst.predict(dtest)
                ypred =numpy.argmax(probs, axis=1)

                

                error = float(numpy.sum(ypred != YTest))
                e += error
                c += float(len(YTest))

            e/=c

            return e



    def save(self, fname):
        self.bst.save_model(fname)

    def load(self, fname):
        self.bst = xgb.Booster({'nthread':self.nThreads}) #init model
        self.bst.load_model(fname)             # load data

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)