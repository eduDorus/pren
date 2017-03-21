#import numpy as np
from picamera import PiCamera
from time import sleep
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



# Load Model
network = input_data(shape=[None, 128, 128, 1])
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 1)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

#model = DNN(network, tensorboard_verbose=0)
#model.load('models/mid_prod_model')

camera = PiCamera()
camera.resolution = (720, 380)

i = 0
while (True):
    i += 1
    camera.capture('images/ampel/1/images_{0}.jpg'.format(i))
    print("captured image : " + "image_{0}.jpg".format(i))
