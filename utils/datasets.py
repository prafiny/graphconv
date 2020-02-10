import numpy as np
import os
import h5py
from keras.utils.data_utils import get_file

path = os.path.dirname(os.path.realpath(__file__))

def mnist_reduced(path='mnist_reduced.npz'):
    path = get_file(path,
                    origin='https://www.dropbox.com/s/fbm842x75nqi9s9/mnist_reduced.npz?raw=1',
                    file_hash='60ae428bd2426466908ed5a4f7d6130e')
    
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_valid, y_valid = f['x_valid'], f['y_valid']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def mnist_reduced_rag(path='mnist_reduced_rag.npz'):
    path = get_file(path,
                    origin='https://www.dropbox.com/s/6079u750d8gnc78/mnist_reduced_rag.npz?raw=1',
                    file_hash='ffac03c6331b1fec57a6be1e29f4716f')
    
    f = np.load(path)
    feat_train, adj_train, y_train = f['feat_train'], f['adj_train'], f['y_train']
    feat_valid, adj_valid, y_valid = f['feat_valid'], f['adj_valid'], f['y_valid']
    feat_test, adj_test, y_test = f['feat_test'], f['adj_test'], f['y_test']
    f.close()
    return (feat_train, adj_train, y_train), (feat_valid, adj_valid, y_valid), (feat_test, adj_test, y_test)

def raw_mnist_reduced_rag(path='raw_mnist_reduced_rag.npz'):
    path = get_file(path,
                    origin='https://www.dropbox.com/s/u0if3zt925wnyqh/raw_mnist_reduced_rag.npz?raw=1',
                    file_hash='4968188f47f40b75a9e65c8c5ff89819')
    
    f = np.load(path)
    feat_train, adj_train, y_train = f['feat_train'], f['adj_train'], f['y_train']
    feat_valid, adj_valid, y_valid = f['feat_valid'], f['adj_valid'], f['y_valid']
    feat_test, adj_test, y_test = f['feat_test'], f['adj_test'], f['y_test']
    f.close()
    return (feat_train, adj_train, y_train), (feat_valid, adj_valid, y_valid), (feat_test, adj_test, y_test)

def mnist_rag(path='mnist_superpixels_data_75.npz'):
    """Loads the MNISTRAG dataset.
    # Returns
        Tuple of Numpy arrays: `(feat_train, adj_train, y_train), (feat_test, adj_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://www.dropbox.com/s/2i19r4cnuzvr895/mnist_superpixels_data_75.npz?raw=1',
                    file_hash='e037a4c05046fca938886ad6a811ac13')
    
    f = np.load(path)
    feat_train, adj_train, y_train = f['feat_train'], f['adj_train'], f['y_train']
    feat_test, adj_test, y_test = f['feat_test'], f['adj_test'], f['y_test']
    f.close()
    return (feat_train, adj_train, y_train), (feat_test, adj_test, y_test)

def raw_mnist_rotated_rag(path='raw_mnist_rotated_rag.npz'):
    """Loads the MNISTRAG dataset.
    # Returns
        Tuple of Numpy arrays: `(feat_train, adj_train, y_train), (feat_test, adj_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://www.dropbox.com/s/w1zagbl7jsdbb9n/raw_mnist_rotated_rag.npz?raw=1',
                    file_hash='1aa6907ed748fa657564644840f70eea')
    
    f = np.load(path)
    feat_train, adj_train, y_train = f['feat_train'], f['adj_train'], f['y_train']
    feat_valid, adj_valid, y_valid = f['feat_valid'], f['adj_valid'], f['y_valid']
    feat_test, adj_test, y_test = f['feat_test'], f['adj_test'], f['y_test']
    f.close()
    return (feat_train, adj_train, y_train), (feat_valid, adj_valid, y_valid), (feat_test, adj_test, y_test)

def mnist_rotated_rag(path='mnist_rotated_rag.npz'):
    """Loads the MNISTRAG dataset.
    # Returns
        Tuple of Numpy arrays: `(feat_train, adj_train, y_train), (feat_test, adj_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://www.dropbox.com/s/pj0mzfq8qfk9cla/mnist_rotated_rag.npz?raw=1',
                    file_hash='2bc7e73d3c553049694f263899ac1b2a')
    
    f = np.load(path)
    feat_train, adj_train, y_train = f['feat_train'], f['adj_train'], f['y_train']
    feat_valid, adj_valid, y_valid = f['feat_valid'], f['adj_valid'], f['y_valid']
    feat_test, adj_test, y_test = f['feat_test'], f['adj_test'], f['y_test']
    f.close()
    return (feat_train, adj_train, y_train), (feat_valid, adj_valid, y_valid), (feat_test, adj_test, y_test)

def mnist_rotated(path='mnist_rotated.npz'):
    """Loads the MNISTRAG dataset.
    # Returns
        Tuple of Numpy arrays: `(feat_train, adj_train, y_train), (feat_test, adj_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://www.dropbox.com/s/jzu137zojgmnmn6/mnist_rotated.npz?raw=1',
                    file_hash='ce533e420cf222621106e7f13a861e25')
    
    f = np.load(path)
    x_train, y_train = np.transpose((f['x_train']*255).astype(np.int), [0, 2, 1]), f['y_train']
    x_valid, y_valid = np.transpose((f['x_valid']*255).astype(np.int), [0, 2, 1]), f['y_valid']
    x_test, y_test = np.transpose((f['x_test']*255).astype(np.int), [0, 2, 1]), f['y_test']
    f.close()
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
