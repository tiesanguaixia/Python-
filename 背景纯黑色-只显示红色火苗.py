
# coding: utf-8

# In[2]:

#采用以下两个库进行视频图像分割与处理
import cv2
import numpy as np
 
#video_file = "video_1.mp4"
video = cv2.VideoCapture('C:\\Users\\Administrator\\Desktop\\huo (1).avi')

framecount = 0
 
while True:
    (grabbed, frame) = video.read()
    framecount+=1
    if not grabbed:
        break
 
    frame = cv2.resize(frame, (320,240))
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
    #lower = [18, 50, 50]
    #upper = [35, 255, 255]
    
    lower = [0, 0, 255]
    upper = [30, 20,255]
    

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)            

    output = cv2.bitwise_and(frame, hsv, mask=mask)

    no_red = cv2.countNonZero(mask)
    cv2.imshow("frame", frame)
    cv2.imshow("output", output)
    #print(no_red)
    #print("output:", frame)
    #if int(no_red) > 20000:
    #    print ('Fire detected')
    #print(int(no_red))
   #print("output:".format(mask))
    key = cv2.waitKey(10)
    if key == 27: # exit on ESC
        break

print(framecount)
cv2.destroyAllWindows()
video.release()