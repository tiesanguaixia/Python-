
# coding: utf-8

import cv2
import numpy as np
import os

width = 640
height = 480

ROI_x = 0
ROI_xmax = width
ROI_y = 0
ROI_ymax = height

color = (234, 148, 33)

video = cv2.VideoCapture( 'C:\\Users\\Administrator\\Desktop\\huo (1).avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

i=1
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    i += 1
    #print(i)
    frame = cv2.resize(frame, (640,480))

    if i>0:
        
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower = [0, 0, 200]
        upper = [35, 30,255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)

        mask[:,:ROI_x] = np.zeros((height, ROI_x), np.int8);
        mask[:,ROI_xmax:] = np.zeros((height, width-ROI_xmax), np.int8);
        mask[:ROI_y,:] = np.zeros((ROI_y, width), np.int8);
        mask[ROI_ymax:,:] = np.zeros((height-ROI_ymax, width), np.int8);

        #cv2.imshow("omask", mask)

        kernel1 = np.ones((3,3),np.uint8)
        kernel2 = np.ones((10,10),np.uint8)
        mask = cv2.erode(mask,kernel1,iterations = 1)
        mask = cv2.dilate(mask,kernel2,iterations = 5)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        output = cv2.bitwise_and(frame, hsv, mask=mask)
        cnt = cv2.findNonZero(mask)
        if cnt is not None:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            fire = cv2.drawContours(frame, [box], 0, color, 2)
            cv2.putText(frame,'Fire Detected!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Fire Detection", fire)

    cv2.imshow("original", frame)
    #out.write(frame)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
video.release()
#out.release()