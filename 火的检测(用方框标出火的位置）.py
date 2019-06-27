
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
    #检查视频是否播放完成，是则grabbed的值为False，循环结束
    if not grabbed:
        break
    i += 1
    #print(i)
    #调整图像大小
    frame = cv2.resize(frame, (640,480))

    if i>0:
        #高斯滤波
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        #转化hsv颜色空间
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower = [0, 0, 200]
        upper = [35, 30,255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        #inRange函数将hsv图像中的介于lower和upper的值变为255，将图像二值化，去除背景
        mask[:,:ROI_x] = np.zeros((height, ROI_x), np.int8);
        mask[:,ROI_xmax:] = np.zeros((height, width-ROI_xmax), np.int8);
        mask[:ROI_y,:] = np.zeros((ROI_y, width), np.int8);
        mask[ROI_ymax:,:] = np.zeros((height-ROI_ymax, width), np.int8);

        #cv2.imshow("omask", mask)

        kernel1 = np.ones((3,3),np.uint8)
        kernel2 = np.ones((10,10),np.uint8)
        mask = cv2.erode(mask,kernel1,iterations = 1)  #腐蚀操作
        mask = cv2.dilate(mask,kernel2,iterations = 5) #膨胀操作
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   开运算，即先腐蚀后膨胀
        output = cv2.bitwise_and(frame, hsv, mask=mask)  #图像按位与
        cnt = cv2.findNonZero(mask)
        if cnt is not None:  #掩膜的点集非空
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
