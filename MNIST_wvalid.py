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
from keras.datasets import mnist as mnist
from keras import backend as K
from utils import graphs
from math import ceil
import tensorflow as tf
from utils.datasets import mnist_reduced, mnist_rotated

# In[2]:

batch_size = 64
l_batch_size = 16
img_cols, img_rows = 14,14
graph_nodes = 14
graph_size = img_cols*img_rows
num_classes = 2
opt = lambda : keras.optimizers.Adam(lr=batch_size/l_batch_size*10**-3)
number_edges_ = 8
n_epochs = 100
cnn = False
train = True
spatial_normalization = False
ks = 9
nh = 1
n_exp = 5

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = mnist_reduced()
#(x_train, y_train), (x_valid, y_valid), _ = mnist_reduced()
#_, _, (x_test, y_test) = mnist_rotated()

def m():
    x_feat = feat_input = Input(shape=feat_shape, name='feat_input')
    x_adj = adj_input = Input(shape=adj_shape, name='adj_input')

    if cnn == True:
        x_feat = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x_feat) 
        x_feat = MaxPooling2D(pool_size=(2, 2))(x_feat) 
        x_feat = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x_feat)
        x_feat = MaxPooling2D(pool_size=(2, 2))(x_feat)
        x_feat = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x_feat) 
        x_feat = MaxPooling2D(pool_size=(2, 2))(x_feat)  
        x = GlobalAveragePooling2D()(x_feat)
    else:
        x_feat = ImageToGraphFeat()(x_feat)
        x_feat = GraphConvNoEdges(kernel_size=ks, filters=32, nhops=nh, activation='relu')([x_feat, x_adj])
        x_feat, x_adj = GraphMaxPooling(reduction_ratio=4)([x_feat, x_adj])
        x_feat = GraphConvNoEdges(kernel_size=ks, filters=64, nhops=nh, activation='relu')([x_feat, x_adj])
        x_feat, x_adj = GraphMaxPooling(reduction_ratio=4)([x_feat, x_adj])
        x_feat = GraphConvNoEdges(kernel_size=ks, filters=256, nhops=nh, activation='relu')([x_feat, x_adj])
        x_feat, x_adj = GraphMaxPooling(reduction_ratio=4)([x_feat, x_adj])
        x = GraphGlobalAveragePooling()([x_feat, x_adj])
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation="softmax", name="predictions")(x)

    # this is the model we will train
    model = Model(inputs=[feat_input, adj_input], outputs=predictions)
    return model

ids_train = y_train < num_classes
ids_valid = y_valid < num_classes
ids_test = y_test < num_classes

y_train = y_train[ids_train]
y_valid = y_valid[ids_valid]
y_test = y_test[ids_test]

x_train = x_train[ids_train]
x_valid = x_valid[ids_valid]
x_test = x_test[ids_test]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

data_format = K.image_data_format()
channel_axis = -1 if data_format == 'channels_last' else 1

x_train = np.expand_dims(x_train, channel_axis)
x_test = np.expand_dims(x_test, channel_axis)
x_valid = np.expand_dims(x_valid, channel_axis)

with tf.Session():
    x_train = tf.image.resize_images(x_train, [img_rows,img_cols]).eval()
    x_test = tf.image.resize_images(x_test, [img_rows,img_cols]).eval()
    x_valid = tf.image.resize_images(x_valid, [img_rows,img_cols]).eval()

if K.image_data_format() == 'channels_first':
    coord = graphs.grid(img_cols).reshape(1, 2, img_rows, img_cols)
else:
    coord = graphs.grid(img_cols).reshape(1, img_rows, img_cols, 2)

#x_train = x_train.reshape(-1, img_rows*img_rows, 1)
#x_test = x_train.reshape(-1, img_rows*img_rows, 1)

#x_train = np.concatenate([x_train, np.tile(coord, [x_train.shape[0], 1, 1, 1])], axis=channel_axis)
#x_valid = np.concatenate([x_valid, np.tile(coord, [x_valid.shape[0], 1, 1, 1])], axis=channel_axis)
#x_test = np.concatenate([x_test, np.tile(coord, [x_test.shape[0], 1, 1, 1])], axis=channel_axis)

# Normalization
axes = 0 if spatial_normalization else (0, 1 if data_format == "channels_last" else -1, 2)

avg = np.mean(x_train, axis=axes, keepdims=True)
stddev = np.std(x_train, axis=axes, keepdims=True)

x_train = (x_train - avg) / (stddev + 10**(-6))
x_valid = (x_valid - avg) / (stddev + 10**(-6))
x_test = (x_test - avg) / (stddev + 10**(-6))

feat_shape = (None, None, x_test.shape[channel_axis])
adj_shape = (None, None)

size_train = y_train.shape[0]
size_valid = y_valid.shape[0]
size_test = y_test.shape[0]

adj = graphs.grid_graph(graph_nodes, number_edges=number_edges_)
adj = adj.toarray()
adj_int = adj.copy()
adj_int[adj_int > 0] = 1.
adj = np.expand_dims(adj_int, 0)
def generator(x, y, adj, batch_size):
    size = x.shape[0]
    arr = np.arange(size)
    while True:
        np.random.shuffle(arr)
        for i in range(0, size, batch_size):
            x_slice = x[arr[i:i+batch_size]]
            y_slice = y[arr[i:i+batch_size]]
            adj_slice = np.tile(adj, [x_slice.shape[0], 1, 1])
            yield ({'feat_input': x_slice, 'adj_input': adj_slice}, y_slice)

# In[6]:
#print(adj)
#print(x_train[0])
#exit()

scores = list()
for i in range(n_exp):
    model = m()
    model.summary()

    # In[ ]:

    if train:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt(),
                      metrics=['accuracy'])

        cb = list()
        cb.append(keras.callbacks.ModelCheckpoint('best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False))
        cb.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', min_lr=10**-20))
        #cb.append(keras.callbacks.TensorBoard(log_dir='/home/administrateur/logs/keras', histogram_freq=0, batch_size=32, write_graph=True))

        res = model.fit_generator(generator(x_train, y_train, adj, batch_size),
                            steps_per_epoch=ceil(size_train/batch_size),
                            epochs=n_epochs,
                            validation_data=generator(x_valid, y_valid, adj, batch_size),
                            validation_steps=ceil(size_valid/batch_size),
                            callbacks=cb
                           )
        print(res.history)

    model.load_weights("best.hdf5", by_name=True)

    if not train:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt(),
                      metrics=['accuracy'])

    # In[ ]:
    scores.append(model.evaluate([feat_test, adj_test], y_test, verbose=0))

for i, score in enumerate(scores):
    print('Exp {}'.format(i))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

print([score[1] for score in scores])
