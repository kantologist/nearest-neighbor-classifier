import numpy as np


class NearestNeighbour(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def predict (self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)
        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr-X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]
        return Ypred
