from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import image_preloader
from sklearn.model_selection import train_test_split

# Load path/class_id image file:
data_folder = 'data/pren_dataset_large'

# Build the preloader array, resize images to 128x128
X, Y = image_preloader(data_folder, image_shape=(128, 128), mode='folder', categorical_labels=True, normalize=True, grayscale=True, files_extension='.jpg')

# Data loading and preprocessing
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
X, Y = shuffle(X, Y)

# Reshape
X = X.reshape([-1, 128, 128, 1])
X_test = np.asarray(X_test).reshape([-1, 128, 128, 1])

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=30.)

# Convolutional network building
network = input_data(shape=[None, 128, 128, 1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 8, 7, strides=3, activation='relu')
network = conv_2d(network, 16, 5, strides=2, activation='relu')
network = conv_2d(network, 32, 3, strides=1, activation='relu')
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
#model.load('model_5')

model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=200, run_id='model_5')

# Save a model
model.save('model_5')
