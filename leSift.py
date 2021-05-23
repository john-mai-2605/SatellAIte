
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('home - Copy.jpg')

gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()

kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(np.shape(img))
cv.imwrite('sift_keypoints.jpg',img)
kp, des = sift.detectAndCompute(gray,None)


np.shape(des)

codebook, distortion = vq.kmeans(des, 3)