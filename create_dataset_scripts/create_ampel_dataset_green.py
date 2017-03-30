from picamera import PiCamera

camera = PiCamera()
camera.resolution = (640, 480)

i = 0
while (True):
    i += 1
    camera.capture('../images/final_dataset/5/images_{0}.jpg'.format(i))
    print("captured image : " + "image_{0}.jpg".format(i))
