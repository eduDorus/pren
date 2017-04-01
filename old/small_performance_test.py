import time
import h5py
import numpy as np
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# Load Model
network = input_data(shape=[None, 128, 128, 1])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

model = DNN(network, tensorboard_verbose=0)
model.load('models/mid_prod_model')


# Load Data
h5f = h5py.File('data/dataset_large.h5', 'r')
X = h5f['X'][:10]
X = np.array(X).reshape([-1, 128, 128, 1])


# Mesure time
start_time = time.time()
for i in range(0, 9):
    predication = model.predict([X[i]])
    print(np.round(predication, 2))

print("--- %s seconds ---" % (time.time() - start_time))
