#from http://answers.opencv.org/question/77046/use-contours-in-a-video/
import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)

    # define range of blue color in HSV
    lower_blue = np.array([95, 105, 20])
    upper_blue = np.array([115, 255, 255])

    # mask for the colors
    mask_blue = cv2.inRange(HSV, lower_blue, upper_blue)

    thresh = cv2.threshold(mask_blue, 127, 255, cv2.THRESH_BINARY)[1]
    #    thresh = cv2.threshold(mask_blue,60,255,cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 5:
        maxcontour = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(maxcontour)
        center = (int(x), int(y))
        a = np.arctan2( x-320, 480-y ) *180/np.pi

        print 'angle: ', a


        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)






    for c in cnts:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)  # draws the Conture lines
        cv2.drawContours(thresh, [c], -1, (0, 255, 0), 1)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('thresh', thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()