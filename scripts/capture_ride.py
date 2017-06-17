from time import sleep
from picamera import PiCamera

# Set run id
RUN_ID = 14

FPS = 5
SLEEP_TIME = 1 / FPS
PATH = '../images/ciffer/new/{0}'.format(RUN_ID)
BOUNDARY = 10 * FPS
CAPTURE_ID = 0

with PiCamera() as camera:
    camera.resolution = (240, 160)
    camera.exposure_mode = 'sports'

    while CAPTURE_ID < BOUNDARY:
        sleep(SLEEP_TIME)
        camera.capture('{0}/{1}_{2}.jpg'.format(PATH, RUN_ID, CAPTURE_ID))
        print(
            'captured image : {0}/{1}_{2}.jpg'.format(PATH, RUN_ID, CAPTURE_ID))
        CAPTURE_ID += 1
