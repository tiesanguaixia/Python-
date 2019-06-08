
# coding: utf-8

# In[1]:


import numpy as np
import cv2
 
#cap = cv2.VideoCapture('D:\\00BUPT\\yan.avi')


knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
camera = cv2.VideoCapture('C:\\Users\\Administrator\\Desktop\\sWasteBasket.avi')
 
def drawCnt(fn, cnt):
  if cv2.contourArea(cnt) > 1400:
    (x, y, w, h) = cv2.boundingRect(cnt)
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
    fg_bgr = cv2.morphologyEx(fg_bgr,cv2.MORPH_OPEN,es)
    bw_and = cv2.bitwise_and(fg_bgr, frame)
    draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)

    #draw = cv2.GaussianBlur(draw, (21, 21), 0)
    draw = cv2.threshold(draw, 10, 255, cv2.THRESH_BINARY)[1]
    ##draw = cv2.dilate(draw, es, iterations = 2)
    #image, contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    #for c in contours:
    #    drawCnt(frame, c)

    cv2.imshow("motion detection", draw)
    cv2.imshow("original", frame)
     
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
  
camera.release()
#cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
video_full_path='C:\\Users\\Administrator\\Desktop\\sWasteBasket.avi'

cap  = cv2.VideoCapture('C:\\Users\\Administrator\\Desktop\\sWasteBasket.avi')

def contrast_brightness_demo(image, c, b):  #其中c为对比度，b为每个像素加上的值（调节亮度）
    blank = np.zeros(image.shape, image.dtype)   #创建一张与原图像大小及通道数都相同的黑色图像
    dst = cv.addWeighted(image, c, blank, 1-c, b) #c为加权值，b为每个像素所加的像素值
    ret, dst = cv.threshold(dst, 25, 255, cv.THRESH_BINARY)
    return dst

redThre = 105
saturationTh = 42
 
camera = cv.VideoCapture('C:\\Users\\Administrator\\Desktop\\sWasteBasket.avi')
width = int(camera.get(3))
height = int(camera.get(4))
 
firstFrame = None

framecount = 0
x = 0
y = 0
w = 0
h = 0
 
while True:
    #print(framecount)
    framecount+=1
    (grabbed, frame) = camera.read()
    if frame is None:
        print("The End!!!")
        break
        
    frame = cv.resize(frame, (320,240))    
    cv.imshow("frame", frame)
    B = frame[:, :, 0]
    G = frame[:, :, 1]
    R = frame[:, :, 2]
    
    minValue = np.array(np.where(R <= G, np.where(G <= B, R, np.where(R <= B, R, B)), np.where(G <= B, G, B)))
    S = 1 - 3.0 * minValue / (R + G + B + 1)
    fireImg = np.array(np.where(R > redThre, np.where(R >= G, np.where(G >= B, np.where(S >= 0.2, np.where(S >= (255 - R)*saturationTh/redThre, 255, 0), 0), 0), 0), 0))
    gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
    gray_fireImg[:, :, 0] = fireImg
    gray_fireImg = cv.GaussianBlur(gray_fireImg, (7, 7), 0)
    gray_fireImg = contrast_brightness_demo(gray_fireImg, 5.0, 25)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    gray_fireImg = cv.morphologyEx(gray_fireImg, cv.MORPH_CLOSE, kernel)
    dst = cv.bitwise_and(frame, frame, mask=gray_fireImg)
    cv.imshow("fire", dst)
    cv.imshow("gray_fireImg", gray_fireImg)
    c = cv.waitKey(40)
    if c == 27:
        break

camera.release()
cv.destroyAllWindows()

#cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2
from skimage.feature import local_binary_pattern 
import matplotlib.pyplot as plt
import time


image_back = cv2.resize(image_back_ori, (320,240))
image_front = cv2.resize(image_front_ori, (320,240))
#print(image.shape)
image_back = cv2.cvtColor(image_back, cv2.COLOR_BGR2GRAY)
image_cur = cv2.cvtColor(image_front, cv2.COLOR_BGR2GRAY)
print(image_back.shape)
print(image_cur.shape)
#plt.imshow(image, plt.cm.gray)
#plt.imshow(image)
#plt.show()


# In[ ]:


theta = 4

def csltp(src):
    dst = np.zeros(src.shape,dtype=src.dtype)
     #print(src.shape[0])
     #print(src.shape[1])
    for i in range(1,src.shape[0]-1):
         for j in range(1,src.shape[1]-1):
                summa = 0; 
                if((int(src[i-1][j-1])-int(src[i+1][j+1]))>theta):
                    summa = summa + 2*1
                elif((int(src[i-1][j-1])-int(src[i+1][j+1]))<(-theta)):
                    summa = summa + 1*1
             
                if((int(src[i-0][j-1])-int(src[i+0][j+1]))>theta):
                    summa = summa + 2*4
                elif((int(src[i-0][j-1])-int(src[i+0][j+1]))<(-theta)):
                     summa = summa + 1*4
                
                if((int(src[i+1][j-1])-int(src[i-1][j+1]))>theta):
                    summa = summa + 2*16
                elif((int(src[i+1][j-1])-int(src[i-1][j+1]))<(-theta)):
                     summa = summa + 1*16
                
                if((int(src[i+1][j-0])-int(src[i-1][j+0]))>theta):
                    summa = summa + 2*64
                elif((int(src[i+1][j-0])-int(src[i-1][j+0]))<(-theta)):
                     summa = summa + 1*64                
                dst[i][j] = summa;  
    return dst

ltp_back = csltp(image_back)
ltp_cur = csltp(image_cur)

plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.imshow(ltp_back, plt.cm.gray)
plt.subplot(212)
plt.imshow(ltp_cur, plt.cm.gray)



# In[ ]:


def divide_method(img,m,n,gx,gy):#分割成m行n列
    h, w = img.shape[0],img.shape[1]
    #gx, gy = np.mgrid(np.linspace(0, w, n+1),np.linspace(0, h, m+1))

    divide_image = np.zeros([m-1, n-1, int(h/m), int(w/n)], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
    #print(divide_image.shape)
    for i in range(m-1):
        for j in range(n-1):      
            #print(i,j)
            #print(gy[i][j],gy[i+1][j+1])
            #print(gx[i][j],gx[i+1][j+1])
            divide_image[i,j,0:gy[i+1][j+1]-gy[i][j], 0:gx[i+1][j+1]-gx[i][j]]= img[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]]
    return divide_image
m=30
n=40
w=320
h=240
gx, gy = np.meshgrid(np.linspace(0, w, n+1),np.linspace(0, h, m+1))
gx=np.floor(gx).astype(np.int)
gy=np.floor(gy).astype(np.int)
divide_image_back = divide_method(ltp_back,m,n,gx,gy)
print(divide_image_back.shape)
divide_image_cur = divide_method(ltp_cur,m,n,gx,gy)


# In[ ]:


tp = 80
ts = 100
block_num = 0

display =  np.zeros(image_cur.shape,dtype=image_cur.dtype)
for i in range(0,image_cur.shape[0]):
    for j in range(0,image_cur.shape[1]):
                display[i][j] = image_cur[i][j]

for i in range(m-1):
    for j in range(n-1):
        block_num+=1
        #plt.imshow(ltp_back, plt.cm.gray)
        #plt.hist(ltp_back)
        #hist_back=np.histogram(ltp_back, bins=8)
        #hist_cur=np.histogram(ltp_current, bins=8)
        #plt.imshow(hist_back, plt.cm.gray)
        fore_det = 0
        smoke_det = 0
        #print(hist_cur[0][0])
        #image_front[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]]=ltp_current
        #image_back[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]]=ltp_cur_hisback
        #print(divide_image_cur[i,j])
        #print(divide_image_back[i,j])
        for k in range(8):
            #for l in range(8):
            cur_his = divide_image_cur[i,j,k].mean()
            back_his = divide_image_back[i,j,k].mean()
            fore_det += abs(int(back_his)-int(cur_his))
                #print(fore_det)
            smoke_det += int(back_his)-int(cur_his)
            #print(hist_back[0][k],hist_cur[0][k])

        if(fore_det>tp and smoke_det>ts):
        #if(fore_det>tp):
            print(i,j,fore_det,smoke_det)
            print(divide_image_cur[i,j])
            print(divide_image_back[i,j])
            #divide_image_back[i,j,:,:] = 255
            #print(block_num,i,j)
            #display[i][j] = 255
            display[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]]=255
            #divide_image[i,j,0:gy[i+1][j+1]-gy[i][j], 0:gx[i+1][j+1]-gx[i][j]]= img[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]]
        #print(hist_back)
        #print(hist_cur)
        #n_cur, bins_cur, patches_cur = plt.hist(divide_image_back[i,j,:,:,:].ravel(),8)

#x = csltp(image)
#print(x[5][316])
#print(x)
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.imshow(display, plt.cm.gray)
plt.subplot(212)
plt.imshow(image_back, plt.cm.gray)
#plt.imsave('D:\\00BUPT\\yan_ltp.png',x,cmap='gray')
#n, bins, patches = plt.hist(x.ravel(),8)

#plt.show()


# In[ ]:


theta = 0.05
alpha = 0.005
Tp = 80
Ts = 100
Tgray = 0.15 # not sure assume
Twh = 0.3#totally have no idea
Tbk2 = 0.1 #totally have no idea
Tbk1 = 0.3 #totally have no idea
count = 0


time_start = time.time()

image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
imagebg = cv2.cvtColor(imagebg_ori, cv2.COLOR_BGR2GRAY) 

imagebg = image * alpha + (1 - alpha) * imagebg

for i in range(0,8):
    for j in range(0,8):
        img_test = image[i*30 : i*30 + 30, j*40 : j*40 + 40]
        lbp = np.zeros((30,40))
        for k in range(1,27):
            for q in range(1,37):
                lbp[k,q] = ltp(img_test[k-1:k+2,q-1:q+2],theta)
        arr=lbp.flatten()
        
        n, bins, patches = plt.hist(arr, bins=16, normed=1, facecolor='red', alpha=0.75) 
        
        
        imgbg_test = imagebg[i*30 : i*30 + 30, j*40 : j*40 + 40]
        lbpbg = np.zeros((30,40))
        for m in range(1,27):
            for n in range(1,37):
                lbpbg[m,n] = ltp(imgbg_test[m-1:m+2,n-1:n+2],theta)
        print(arr1)
        arr1=lbpbg.flatten()
        n, bins1, patches = plt.hist(arr1, bins=16, normed=1, facecolor='red', alpha=0.75) 
        
        diff = np.sum(np.abs(bins1 - bins),axis =0)
        
        if diff  >=  Tp:
            if diff < Ts:
                Cmin = np.min(image_ori[i*30 : i*30 + 30, j*40 : j*40 + 40],axis = 2)
                Cmax = np.max(image_ori[i*30 : i*30 + 30, j*40 : j*40 + 40],axis = 2)
                Cdiff = np.abs(Cmax - Cmin)
                Cdiff[Cdiff <Tgray] = 0.1

                Ifg = img_test
                Ibg = imgbg_test
                Idiff = (Ifg - Ibg)
                Idiff1 = Idiff
                print(Idiff1)
                Idiff[Idiff > Twh] = 0.2
                Idiff[(Idiff > Tbk2) & (Idiff < Tbk1)] = 0.3
                
                image[(Cdiff == 0.1) & ((Idiff == 0.2)|(Idiff == 0.3))] = 255
                print("possible")
        else:
            print("background")
plt.imshow(image)
plt.show()
        
time_end = time.time()
print('time: ', time_end - time_start) # 以秒为单位  


# In[ ]:


lbp = local_binary_pattern(image, 8, 1)
plt.subplot(111)
plt.imshow(lbp, plt.cm.gray)


# In[ ]:


def ltp(arr,theta):
    arr1 = arr.flatten()
    summa = 0
    for i in range(0,4):
        j = 8 - i
        summa += (cs((arr1[i] - arr1[j]),theta)) * (2**(2*i))
    return summa
def cs(num,theta):
    if num > theta:
        return 2
    elif np.abs(num) <= theta:
        return 0
    else:
        return 1


# In[ ]:


image = image[0:30,0:40]
plt.imshow(image),plt.show()
lbp = local_binary_pattern(image, 8, 3)
plt.imshow(lbp),plt.show()
arr=lbp.flatten()
n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='red', alpha=0.75) 


# In[ ]:


imgbg = cv2.imread("video_900.jpg")
imgbg = cv2.cvtColor(imgbg, cv2.COLOR_BGR2GRAY)
# lbpbg = imgbg[:30,:40]
lbpbg = local_binary_pattern(imgbg, 8, 1)
plt.imshow(lbpbg),plt.show()
arr1=lbpbg.flatten()
n, bins, patches = plt.hist(arr1, bins=8, normed=1, facecolor='red', alpha=0.75) 


# In[ ]:


bins


# In[ ]:


image_ori = cv2.imread("video_2.jpg")
imagebg = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)

for i in range(0,8):
    for j in range(0,8):
        imgbg_test = imagebg[i*30 : i*30 + 30, j*40 : j*40 + 40]
        lbpbg = np.zeros((30,40))
        for m in range(1,29):
            for n in range(1,39):
                lbpbg[m,n] = ltp(imgbg_test[m-1:m+2,n-1:n+2],0.05)


# In[ ]:


lbpbg

