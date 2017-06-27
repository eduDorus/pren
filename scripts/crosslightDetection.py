import display
display.eight()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

from picamera import PiCamera, array
from time import strftime, sleep, time
from numpy import around
import os
import sys


def loadCrosslightModel(image_height, image_width):

    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(image_height, image_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.load_weights('/home/pi/repositories/pren/models/crosslight.h5')

    return model


def loadCifferModel(image_height, image_width):

    model = Sequential()
    model.add(Conv2D(16, (7, 7), activation='relu', padding='same', name='block1_conv1', input_shape=(image_height, image_width, 1)))
    model.add(Conv2D(16, (7, 7), activation='relu', padding='same', name='block1_conv2',))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='block1_pool'))

    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', name='block2_conv1',))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', name='block2_conv2',))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='block2_pool'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1',))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2',))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='block3_pool'))

    # the model so far outputs 3D feature maps (height, width, features)

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="fc-1"))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu', name="fc-2"))
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.load_weights('/home/pi/repositories/pren/models/ciffer.h5')
    
    return model


def crosslightDetection(image_height, image_width, model):
    with PiCamera() as camera:
        camera.resolution = (image_height, image_width)
        sleep(1)
        start_time = 0
        red_counter = 0

        with array.PiRGBArray(camera) as output:
            while True:
                camera.capture(output, 'rgb')

                x = output.array
                x = x.reshape((1,) + x.shape)
                x = x * (1. / 255)

                prediction = around(model.predict(x), 3)[0][0]
                #print(prediction)

                if prediction > 0.80:
                    red_counter += 1

                if red_counter == 5:
                    display.zero()
                    start_time = time()

                if ((time() - start_time) > 60):
                    print("0")
                    sys.stdout.flush()
                    break

                if (prediction < 0.20 and red_counter > 5):
                    print("0")
                    sys.stdout.flush()
                    break

                output.truncate(0)

def captureContinouse(image_height, image_width, timeInSeconds):

    start_time = time()
    CAPTURE_ID = 0

    directory = '/home/pi/repositories/pren/images/ciffer/run/{0}'.format(strftime("%d%m%Y-%H%M"))

    if not os.path.exists(directory):
        os.makedirs(directory)

    with PiCamera() as camera:
        camera.resolution = (image_width, image_height)
        camera.exposure_mode = 'sports'

        while True:
            sleep(0.2)
            camera.capture(directory + '/image{0}_{1}.jpg'.format(strftime("%H%M%s"), CAPTURE_ID))
            CAPTURE_ID += 1

            if timeInSeconds < (time() - start_time):
                break

    # with PiCamera() as camera:
    #     camera.exposure_mode = 'sports'
    #     camera.resolution = (image_width, image_height)
    #     sleep(0.2)

    #     for i, filename in enumerate(camera.capture_continuous(directory + '/image{counter:03d}.jpg')):
    #         sleep(0.2)
    #         if i >= count:
    #             break

    return directory

def predictCiffer(directory):
    display.four()
    print("4")
    sys.stdout.flush()

def logic():

    # Load Models
    crosslightModel = loadCrosslightModel(128, 128)
    cifferModel = loadCifferModel(256, 256)

    # Detect Crosslight
    crosslightDetection(128, 128, crosslightModel)

    # Capture Ride
    directory = captureContinouse(256, 256, 37)

    # Make Prediction
    predictCiffer(directory)

logic()