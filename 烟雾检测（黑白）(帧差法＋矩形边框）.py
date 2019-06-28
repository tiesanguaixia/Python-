# coding: utf-8

#采用以下两个库进行视频图像分割与处理
import numpy as np
import cv2
#OpenCV中背景减除方法中的KNN算法，专门用来解决移动目标跟踪问题。
knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)
#使用getStructuringElement定义一个3*3椭圆形结构元素。
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
camera = cv2.VideoCapture('C:\\Users\\Administrator\\Desktop\\sWasteBasket.avi')


#注意：这里是另一种画边框来包围烟雾区域的方法的核心函数！
def drawCnt(fn, cnt):
  #先进行轮廓面积的计算和判断
  if cv2.contourArea(cnt) > 1400:
    (x, y, w, h) = cv2.boundingRect(cnt)
    #在原图fn的基础上根据左上角的点的坐标，宽和高画出一个包围原图形的矩形边框
    cv2.rectangle(fn, (x, y), (x + w, y + h), (255, 255, 0), 2)


framecount=0
while True:
    ret, frame = camera.read()
    framecount+=1
    if not ret:
        break
    frame = cv2.resize(frame, (320,240))
    frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    fg = knn.apply(frame.copy())
    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    #进行开运算，指的是先进行腐蚀操作，再进行膨胀操作。
    fg_bgr = cv2.morphologyEx(fg_bgr,cv2.MORPH_OPEN,es)
    bw_and = cv2.bitwise_and(fg_bgr, frame)
    draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)
    draw = cv2.threshold(draw, 10, 255, cv2.THRESH_BINARY)[1]
    
    #调用drawCnt函数进行画矩形边框的操作
    contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    for c in contours:
        drawCnt(frame, c)

    #下面开始在窗口中显示图像。
    cv2.imshow("motion detection", draw)
    cv2.imshow("original", frame)
     
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
  
camera.release()
#cap.release()
cv2.destroyAllWindows()