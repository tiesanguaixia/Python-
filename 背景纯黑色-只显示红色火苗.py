# coding: utf-8

#采用以下两个库进行视频图像分割与处理
import cv2
import numpy as np
 
#参数可以是数字，对应摄像头编号，也可以是视频绝对路径。
video = cv2.VideoCapture('C:\\Users\\Administrator\\Desktop\\huo (1).avi')
framecount = 0
while True:
    (grabbed, frame) = video.read()
    framecount+=1
    if not grabbed:
        break
    #调整图像大小
    frame = cv2.resize(frame, (640,480))
    #高斯平滑滤波：参数分别是图像，滤波器大小，标准差。高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大。
    #这种图像模糊处理技术平滑效果柔和，而且边缘保留的也较好。
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    #图像颜色转换函数：转换为HSV颜色空间，这个模型中颜色的参数分别是：色调（H），饱和度（S），明度（V）。
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
    lower = [0, 0, 255]
    upper = [30, 20,255]
    
    #uint8表示8位无符号整型数组。
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    #下面设置阈值，去除背景部分，将低于lower和高于upper的部分图像值分别变成0，lower～upper之间的值变成255。
    mask = cv2.inRange(hsv, lower, upper)            
    #执行图像的像素值按位与运算，用掩膜实现图像的先遮挡后叠加。
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)
    #下面开始在窗口中显示图像。
    cv2.imshow("frame", frame)
    cv2.imshow("output", output)
    key = cv2.waitKey(10)
    if key == 27: # exit on ESC
        break

print(framecount)
cv2.destroyAllWindows()
video.release()