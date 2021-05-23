
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



class siftBuilder():
    def __init__(self):
        self.images=[]
        self.descriptors = []

    def add_image(self, im_arr):
        for i in im_arr:
            self.images.append(i)

    def sift(self):
        descriptors=[]
        for i in self.images:
            kp,des = sift.detetAndCompute(i)
            descriptors.append(des)
        return np.array(self.descriptors)


if __name__ == "__main__":
    img = cv.imread('urban.jpg')

    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()

    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(np.shape(img))
    cv.imwrite('sift_keys.jpg',img)
    kp, des = sift.detectAndCompute(gray,None)


    np.shape(des)

    # codebook, distortion = vq.kmeans(des, 3)