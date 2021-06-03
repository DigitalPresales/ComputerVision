import cv2
import numpy as np 

#myimage = cv2.imread(r'File path of golengate.jpg',0)

myimage = cv2.imread(r'File path of p2.jpg',0)
# here we can use data ingestion layer to pull in streaming or event based data from source system
# to handle huge data volume, we can use kafka or AWS kinesis

height, width = myimage.shape

ms_x = cv2.Sobel(myimage, cv2.CV_64F,0,1,ksize=5)
ms_y = cv2.Sobel(myimage, cv2.CV_64F,1,0,ksize= 5)

cv2.imshow('Original picture', myimage)
cv2.waitKey(0)
cv2.imshow('Sobel X', ms_x)
cv2.waitKey(0)
cv2.imshow('Sobel Y', ms_y)
cv2.waitKey(0)

ms_OR = cv2.bitwise_or(ms_x, ms_y)
cv2.imshow('Sobel _OR operation ', ms_OR)
cv2.waitKey(0)

lap = cv2.Laplacian(myimage, cv2.CV_64F)
cv2.imshow('Laplacian data', lap)
cv2.waitKey(0)

can = cv2.Canny(myimage, 50,120)
cv2.imshow('canny', can)
cv2.waitKey(0)

#here we can reduce the noise from the picture and send that image to next processing laye
# this data can also be fed to database to enable analytics 

cv2.destoryAllWindows()
