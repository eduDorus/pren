import display
display.five()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

from picamera import PiCamera, array
from time import strftime, sleep
from numpy import around
import os


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
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(image_height, image_width, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2',))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1',))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2',))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1',))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv2',))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='block3_pool'))

    model.add(Flatten())
    model.add(Dense(96, activation='relu', name="dense1"))
    model.add(Dropout(1))
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

        with array.PiRGBArray(camera) as output:
            while True:
                camera.capture(output, 'rgb')

                x = output.array
                x = x.reshape((1,) + x.shape)
                x = x * (1. / 255)

                prediction = around(model.predict(x), 3)[0][0]

                display.zero()

                if prediction < 0.10:
                    print("go")
                    break

                output.truncate(0)

def captureContinouse(timeInSeconds):

    count = timeInSeconds * 5

    directory = '../images/ciffer/run/{0}'.format(strftime("%d%m%Y-%H%M"))

    if not os.path.exists(directory):
        os.makedirs(directory)

    with picamera.PiCamera() as camera:
        camera.exposure_mode = 'sports'
        camera.resolution = (image_height, image_width)
        sleep(0.5)

        for i, filename in enumerate(camera.capture_continuous(directory + '/image{counter:03d}.jpg')):
            time.sleep(0.2)
            if i == count:
                break

    return directory

def predictCiffer(directory):
    test_datagen = ImageDataGenerator(rescale=1./255)


def logic():

    # Load Models
    crosslightModel = loadCrosslightModel(128, 128)
    cifferModel = loadCifferModel(160, 240)

    # Detect Crosslight
    crosslightDetection(128, 128, crosslightModel)

    # Capture Ride
    directory = captureContinouse(50)

    # Make Prediction

logic()