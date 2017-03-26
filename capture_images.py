import picamera
from time import sleep

camera = picamera.PiCamera()

for i in range(5):
    sleep(5)
    camera.capture('/home/pi/images/image%s.jpg' % i)
