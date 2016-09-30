import numpy as np
from load_data import load_CIFAR10
from nearest_neighbour import NearestNeighbour

Xtr,Ytr,Xte,Yte = load_CIFAR10('data/cifar10/')

Xtr_rows = Xtr.reshape(Xtr.shape[0],32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0],32*32*3)

nn = NearestNeighbour()
nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)
print ('accuracy: %f' %( np.mean(Yte_predict == Yte)))
