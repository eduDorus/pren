from picamera import PiCamera

camera = PiCamera()
camera.resolution = (480, 320)

i = 0
while (True):
    i += 1
    camera.capture('images/ampel/3/images_{0}.jpg'.format(i))
    print("captured image : " + "image_{0}.jpg".format(i))
