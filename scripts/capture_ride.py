import time
from picamera import PiCamera

RUN_ID = 0
RUN_TIME = 50
PATH = 'run/${0}'.format(RUN_ID)
CAPTURE_ID = 0
SLEEP_TIME = 0.2

t_end = time.time() + RUN_TIME
with PiCamera() as camera:
    camera.resolution = (320, 480)
    camera.exposure_mode = 'sport'
    #camera.awb_mode = 'fluorescent'

    sleep(1)

    while time.time() < t_end:
        camera.capture('{0}/{1}_{2}.jpg'.format(PATH, RUN_ID, CAPTURE_ID))
        print('captured image : {0}/{1}_{2}.jpg'.format(PATH, RUN_ID, CAPTURE_ID))
        
        CAPTURE_ID += 1
        sleep(SLEEP_TIME)
