#%%
# Import time for performance measurements
import numpy as np
from picamera import PiCamera, array
from time import time, sleep
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Start time measuring
START_TIME = time()

# Model URL
MODEL_URL = '../models/final_model_4_160/final_model_4_160'

# Real-time data augmentation (This is only used while training the DNN)
img_aug = ImageAugmentation()
img_aug.add_random_blur(0.25)
img_aug.add_random_rotation(max_angle=10.0)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
print("--- {0} seconds for image preprocessing/augmentation functions ---".format(round(time() - START_TIME)))

# Convolutional network building
network = input_data(shape=[None, 120, 160, 3],
                     data_augmentation=img_aug,
                     data_preprocessing=img_prep)
network = conv_2d(network, 16, 7, activation='relu', name="conv2d-1")
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 5, activation='relu', name="conv2d-2")
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu', name="conv2d-3")
network = max_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 7, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Create the model
model = DNN(network, tensorboard_verbose=0)
print("--- {0} seconds for creating the cnn model ---".format(round(time() - START_TIME)))

# Load the model
model.load(MODEL_URL)
print("--- {0} seconds for loading the model ---".format(round(time() - START_TIME)))

#%%
# Capture images for prediction
with PiCamera() as camera:
    camera.resolution = (160, 120)
    sleep(2)
    with array.PiRGBArray(camera) as output:
        while True:
            camera.capture(output, 'rgb')

            # Make prediction
            prediction_time = time()
            prediction = model.predict(
                output.array.astype(float).reshape((1, 120, 160, 3)))
            print(np.round(prediction, 2))

            print(
                "--- {0} seconds a prediction ---".format(time() - prediction_time))
            output.truncate(0)
