
# coding: utf-8

# # Production ready crosslight detection

# ### Lets create our model and load the weights

# In[5]:

import display
display.five()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from picamera import PiCamera, array
from time import sleep
from numpy import around

image_height = 128
image_width = 128

display.four()


# In[6]:

model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=(image_height, image_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
display.three()

model.load_weights('/home/pi/repositories/pren/models/crosslight.h5')

display.two()


# ### Raspberry camera logic

# In[16]:

with PiCamera() as camera:
    camera.resolution = (image_height, image_width)
    
    # Sleep to let camera adjust to the light
    sleep(1)
    
    #display.one()
    with array.PiRGBArray(camera) as output:
        while True:
            camera.capture(output, 'rgb')
            
            x = output.array
            x = x.reshape((1,) + x.shape)
            x = x * (1./255)
            
            prediction = around(model.predict(x), 3)[0][0]
            
            display.zero()
            
            if prediction < 0.05:
                print("go")
                break
            
            output.truncate(0)

