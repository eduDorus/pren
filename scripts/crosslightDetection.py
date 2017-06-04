
# coding: utf-8

# In[ ]:

# Imports
import display
display.five()

from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
display.four()

# In[ ]:

# Neural Net
network = input_data(shape=[None, 224, 224, 3])
network = conv_2d(network, 32, 5, activation='relu', name="conv2d-1")
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu', name="conv2d-2")
network = max_pool_2d(network, 2)
network = fully_connected(network, 96, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 96, activation='relu')
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = DNN(network, tensorboard_verbose=0)
display.three()


# In[ ]:

model.load('/home/pi/repositories/pren/models/model_crosslight')
display.two()

# In[ ]:

from picamera import PiCamera, array
from time import sleep


# In[ ]:

with PiCamera() as camera:
    camera.resolution = (224, 224)
    sleep(1)
    counter = 0
    display.one()
    with array.PiRGBArray(camera) as output:
        while counter < 1:
            camera.capture(output, 'rgb')
            prediction = np.around(model.predict(output.array.astype(float).reshape((1, 224, 224, 3))), 3)
            green = prediction[0]
            # print(green[0])
            if green[1] > 0.90:
                display.zero()
            if green[0] > 0.90:
                counter += 1
            output.truncate(0)
        print("go")


