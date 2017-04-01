import time
start_time = time.time()
import picamera
import picamera.array
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

# Time for imports
print("--- %s seconds for imports ---" % (time.time() - start_time))
start_time = time.time()

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Load Model
network = input_data(shape=[None, 640, 480, 3],
                     data_preprocessing=img_prep)
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 1)
network = fully_connected(network, 8, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

#model = DNN(network, tensorboard_verbose=0)
#model.load('models/pren_model')

# Time to create the model
print("--- %s seconds for loading model ---" % (time.time() - start_time))
start_time = time.time()

# Capture images for prediction
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as output:
        camera.resolution = (640, 480)
        camera.capture(output, 'rgb')
        print('Captured %dx%dx%d image' % (
                output.array.shape[1], output.array.shape[0], output.array.shape[2]))
        print(output.array.shape)
        model.predict(output.array)
        output.truncate(0)
        
# Time for imports
print("--- %s seconds for capture one image with prediction ---" % (time.time() - start_time))
start_time = time.time()
