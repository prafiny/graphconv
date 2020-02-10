
# coding: utf-8

# In[1]:


import keras
from utils import layers
layers.setup()
from utils.layers import GraphConvNoEdges, GraphMaxPooling, GraphGlobalAveragePooling, ImageToGraphFeat
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import backend as K
from utils import graphs
from math import ceil
import tensorflow as tf
from utils.datasets import raw_mnist_reduced_rag, raw_mnist_rotated_rag

dataset = raw_mnist_reduced_rag

# In[2]:
n_exp = 1
batch_size = 256
l_batch_size = 16
#img_cols, img_rows = 14,14
#graph_nodes = 14
#graph_size = img_cols*img_rows
num_classes = 443
opt = lambda : keras.optimizers.Adam(lr=batch_size/l_batch_size*10**-3)
number_edges_ = 8
n_epochs = 100
train = True
mixed = False
ks = 9 # avg(2hops) = 16.89, std(2hops) = 5.41, avg+std(2hops) = 23
nh = 1

def m():
    x_feat = feat_input = Input(shape=feat_shape, name='feat_input')
    x_adj = adj_input = Input(shape=adj_shape, name='adj_input')
    act = 'relu'
#    x_feat = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x_feat) 
#    x_feat = MaxPooling2D(pool_size=(2, 2))(x_feat) 
#    x_feat = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x_feat)
#    x_feat = MaxPooling2D(pool_size=(2, 2))(x_feat)
#    x_feat = ImageToGraphFeat()(x_feat)
    #x_feat, x_adj = GraphMaxPooling(reduction_ratio=2)([x_feat, x_adj])
    x_feat = GraphConvNoEdges(kernel_size=ks, filters=32, nhops=nh, activation=act)([x_feat, x_adj])
    x_feat, x_adj = GraphMaxPooling(reduction_ratio=2)([x_feat, x_adj])
    x_feat = GraphConvNoEdges(kernel_size=ks, filters=64, nhops=nh, activation=act)([x_feat, x_adj])
    x_feat, x_adj = GraphMaxPooling(reduction_ratio=2)([x_feat, x_adj])
    x_feat = GraphConvNoEdges(kernel_size=ks, filters=256, nhops=nh, activation=act)([x_feat, x_adj])
    x_feat, x_adj = GraphMaxPooling(reduction_ratio=2)([x_feat, x_adj])
    x = GraphGlobalAveragePooling()([x_feat, x_adj])
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation="softmax", name="predictions")(x)

    # this is the model we will train
    model = Model(inputs=[feat_input, adj_input], outputs=predictions)
    return model
if mixed:
    (feat_train, adj_train, y_train), (feat_valid, adj_valid, y_valid), _ = raw_mnist_reduced_rag()
    _, _, (feat_test, adj_test, y_test) = raw_mnist_rotated_rag()
else:
    (feat_train, adj_train, y_train), (feat_valid, adj_valid, y_valid), (feat_test, adj_test, y_test) = dataset()
ids_train = y_train < num_classes
y_train = y_train[ids_train]
y_train = keras.utils.to_categorical(y_train, num_classes)
feat_train = feat_train[ids_train,...]
adj_train = adj_train[ids_train,...]

ids_valid = y_valid < num_classes
y_valid = y_valid[ids_valid]
y_valid = keras.utils.to_categorical(y_valid, num_classes)
feat_valid = feat_valid[ids_valid,...]
adj_valid = adj_valid[ids_valid,...]

ids_test = y_test < num_classes
y_test = y_test[ids_test]
y_test = keras.utils.to_categorical(y_test, num_classes)
adj_test = adj_test[ids_test,...]
feat_test = feat_test[ids_test,...]

data_format = K.image_data_format()
channel_axis = -1 if data_format == 'channels_last' else 1

if len(feat_train.shape) < 3:
    feat_train = np.expand_dims(feat_train, channel_axis)
    feat_valid = np.expand_dims(feat_valid, channel_axis)
    feat_test = np.expand_dims(feat_test, channel_axis)

#with tf.Session():
#    x_train = tf.image.resize_images(x_train, [img_rows,img_cols]).eval()
#    x_test = tf.image.resize_images(x_test, [img_rows,img_cols]).eval()

#if K.image_data_format() == 'channels_first':
#    coord = graphs.grid(img_cols).reshape(1, 2, img_rows, img_cols)
#else:
#    coord = graphs.grid(img_cols).reshape(1, img_rows, img_cols, 2)

#x_train = x_train.reshape(-1, img_rows*img_rows, 1)
#x_test = x_train.reshape(-1, img_rows*img_rows, 1)

#x_train = np.concatenate([x_train, np.tile(coord, [x_train.shape[0], 1, 1, 1])], axis=channel_axis)
#x_test = np.concatenate([x_test, np.tile(coord, [x_test.shape[0], 1, 1, 1])], axis=channel_axis)

# Normalization
nb_channels = feat_train.shape[channel_axis]
tr = np.expand_dims(np.diagonal(adj_train, axis1=1, axis2=2).astype(np.int), axis=channel_axis)
print(tr.shape)
nb_nodes = tr.sum()

reshap = [1, 1, nb_channels] if data_format == 'channels_last' else [1, nb_channels, 1]
weights = np.tile(tr, reshap)
axes = (0, 1) if data_format == 'channels_last' else (0, -1)
avg = np.average(feat_train, weights=weights, axis=axes)
stddev = np.sqrt(np.average((feat_train-avg)**2, weights=weights, axis=axes) * nb_nodes / (nb_nodes - 1))

avg = np.reshape(avg, reshap)
stddev = np.reshape(stddev, reshap)

print(avg.shape)
print(stddev.shape)

feat_train = (feat_train - avg) / (stddev + 10**(-6))
feat_valid = (feat_valid - avg) / (stddev + 10**(-6))
feat_test = (feat_test - avg) / (stddev + 10**(-6))

#ds = np.arange(y_train.shape[0])
#np.random.shuffle(ds)
#feat_valid, adj_valid, y_valid = feat_train[ds[:10000]], adj_train[ds[:10000]], y_train[ds[:10000]]
#feat_train, adj_train, y_train = feat_train[ds[10000:]], adj_train[ds[10000:]], y_train[ds[10000:]]

feat_shape = (None, feat_test.shape[channel_axis])
adj_shape = (None, None)
size_train = y_train.shape[0]
size_valid = y_valid.shape[0]
size_test = y_test.shape[0]

#adj_shape = (graph_nodes**2, graph_nodes**2)

# In[4]:


#adj = graphs.grid_graph(graph_nodes, number_edges=number_edges_)
#adj = adj.toarray()
#adj_int = adj.copy()
#adj_int[adj_int > 0] = 1.
#adj = np.expand_dims(adj_int, 0)

scores = list()
for i in range(n_exp):
    def generator(feat, adj, y, batch_size):
        size = feat.shape[0]
        arr = np.arange(size)
        while True:
            np.random.shuffle(arr)
            for i in range(0, size, batch_size):
                feat_slice = feat[arr[i:i+batch_size]]
                adj_slice = adj[arr[i:i+batch_size]]
                y_slice = y[arr[i:i+batch_size]]
                #adj_slice = np.tile(adj, [x_slice.shape[0], 1, 1])
                yield ({'feat_input': feat_slice, 'adj_input': adj_slice}, y_slice)
    # In[ ]:
    # In[6]:
    #print(adj)
    #print(x_train[0])
    #exit()
    model = m()
    model.summary()
    if train:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])

        cb = list()
        cb.append(keras.callbacks.ModelCheckpoint('best{}.hdf5'.format(i), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False))
        cb.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', min_lr=10**-20))
        #cb.append(keras.callbacks.TensorBoard(log_dir='/home/administrateur/logs/keras', histogram_freq=0, batch_size=32, write_graph=True))

        res = model.fit_generator(generator(feat_train, adj_train, y_train, batch_size),
                            steps_per_epoch=ceil(size_train/batch_size),
                            epochs=n_epochs,
                            validation_data=([feat_valid, adj_valid], y_valid),
                            validation_steps=ceil(size_valid/batch_size),
                            callbacks=cb
                           )

        print(res.history)

    model.load_weights("best{}.hdf5".format(i))

    if not train:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt(),
                      metrics=['accuracy'])
        
    scores.append(model.evaluate([feat_test, adj_test], y_test, verbose=0))

for i, score in enumerate(scores):
    print('Exp {}'.format(i))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test accuracy_topk:', score[2])

print(scores)
