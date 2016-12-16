import cv2

cap = cv2.VideoCapture(0)

#Scale the image to 128 x 128 and set framerate to 10
cap.set(3, 128)
cap.set(4, 128)
cap.set(5, 10)
i = 0

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    # Save image to dataset (root/feature number/image with number)
    name = 'pren_dataset/4/CLASS0_IMG%s.jpg' % i
    cv2.imwrite(name, gray)
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
