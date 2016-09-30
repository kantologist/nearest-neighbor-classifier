import numpy as np

def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def load_CIFAR10(file):
    dict = unpickle(file + 'data_batch_1')
    X = dict.get('data')
    Y =dict.get('labels')
    for files in [file+'data_batch_2', file+'data_batch_3', file+'data_batch_4', file+'data_batch_5']:
        new_dict = unpickle(files)
        new_X = new_dict.get('data')
        new_Y = new_dict.get('labels')
        X=np.concatenate((X,new_X), axis=0)
        Y=np.concatenate((Y,new_Y), axis=0)
    test_dict = unpickle(file+'test_batch')
    X_test = test_dict.get('data')
    Y_test = test_dict.get('labels')
    return X,Y,X_test,Y_test
