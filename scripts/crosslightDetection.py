import display
display.eight()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

from picamera import PiCamera, array
from time import strftime, sleep, time
import numpy as np
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

    #256

    model = Sequential()
    model.add(Conv2D(16, (7, 7), activation='relu', padding='same', name='block1_conv1', input_shape=(image_height, image_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(16, (5, 5), activation='relu', padding='same', name='block4_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', name='block2_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="fc-1-1"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name="fc-2-1"))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    model.load_weights('/home/pi/repositories/pren/models/ciffer.h5')
    
    return model


def crosslightDetection(image_height, image_width, model):
    with PiCamera() as camera:
        camera.resolution = (image_height, image_width)
        camera.framerate = 3
        sleep(1)
        
        start_time = 0
        red_counter = 0

        with array.PiRGBArray(camera) as output:
            while True:
                camera.capture(output, 'rgb', use_video_port=True)

                x = output.array
                x = x.reshape((1,) + x.shape)
                x = x * (1. / 255)

                prediction = np.around(model.predict(x), 3)[0][0]
                #print(prediction)

                if prediction > 0.80:
                    red_counter += 1

                if red_counter == 5:
                    display.zero()
                    start_time = time()

                if time() - start_time > 15 and red_counter > 5:
                    print("0")
                    sys.stdout.flush()
                    break

                if (prediction < 0.20 and red_counter > 5):
                    print("0")
                    sys.stdout.flush()
                    break

                output.truncate(0)


def cifferDetection(image_height, image_width, image_queue):

    start_time = time()
    
    with PiCamera() as camera:

        camera.resolution = (image_height, image_width)
        camera.framerate = 5
        camera.exposure_mode = 'sports'

        with array.PiRGBArray(camera) as stream:

            # Warum Up for Camera'
            sleep(0.5)

            while True:
                capture_time = time()
                camera.capture(stream, 'rgb', use_video_port=True)

                # Process array for CNN prediction
                x = stream.array
                x = x.reshape((1,) + x.shape)
                x = x * (1. / 255)

                image_queue.append(x)

                stream.truncate(0)

                if time() - start_time > 38:
                    break
                


def predictCiffer(image_queue, model):
    while True:

        if not len(image_queue):
            display.three()
            print("3")
            sys.stdout.flush()
            break

        x = image_queue.pop()
        prediction = np.around(model.predict(x), 2)[0]
        #print(prediction)

        if prediction[0] > 0.95:
            display.one()
            print("1")
            sys.stdout.flush()
            break

        if prediction[1] > 0.95:
            display.two()
            print("2")
            sys.stdout.flush()
            break

        if prediction[2] > 0.95:
            display.three()
            print("3")
            sys.stdout.flush()
            break

        if prediction[3] > 0.95:
            display.four()
            print("4")
            sys.stdout.flush()
            break

        if prediction[4] > 0.95:
            display.five()
            print("5")
            sys.stdout.flush()
            break

def logic():

    # Load Models
    crosslightModel = loadCrosslightModel(128, 128)
    cifferModel = loadCifferModel(128, 128)

    # Detect Crosslight
    crosslightDetection(128, 128, crosslightModel)

    # predict Ciffer
    image_queue = []
    cifferDetection(128, 128, image_queue)
    predictCiffer(image_queue, cifferModel)

logic()