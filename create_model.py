import h5py
import numpy as np
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

h5f = h5py.File('data/dataset_large.h5', 'r')
X = h5f['X']
Y = h5f['Y']
X = np.array(X).reshape([-1, 128, 128, 1])

network = input_data(shape=[None, 128, 128, 1])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

model = DNN(network, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=5, shuffle=True, validation_set=0.2, show_metric=True, batch_size=128,
          validation_batch_size=128, run_id='small_model_prod')
model.save('models/small_prod_model')

h5f.close()
