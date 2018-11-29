import numpy as np
import datetime
import time
import cv2
class myPixel:
        a=0
        b=0
        c=0

device = 0
cap = cv2.VideoCapture(device)
# if capture failed to open, try again
if not cap.isOpened():
    cap.open(device)

# only attempt to read if it is opened
if cap.isOpened:
    while True:
        re, img = cap.read()
        # Only display the image if it is not empty
        if re:
                
                                
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # define range of blue color in HSV
                lower_blue = np.array([110,50,50])
                upper_blue = np.array([130,255,255])

                # Threshold the HSV image to get only blue colors
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                res = cv2.bitwise_and(img,img, mask= mask)

                cv2.imshow("video output", img)
                k2 = cv2.waitKey(10) & 0xFF
                if k2 == 97:
                        timetup = time.gmtime()
                        now = time.strftime('%Y %m %d %H %M %S', timetup)
                        cv2.imwrite( "images\\rock\\"+now+".jpg", img)
                        print("Rock taken")
                if k2 == 115:
                        timetup = time.gmtime()
                        now = time.strftime('%Y %m %d %H %M %S', timetup)
                        cv2.imwrite( "images\\paper\\"+now+".jpg", img)
                        print("Paper taken")
                if k2 == 100:
                        timetup = time.gmtime()
                        now = time.strftime('%Y %m %d %H %M %S', timetup)
                        cv2.imwrite( "images\\scissors\\"+now+".jpg", img)
                        print("Scissors taken")
         
        # if it is empty abort
        else:
            print("Error reading capture device")
            break
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Failed to open capture device")
