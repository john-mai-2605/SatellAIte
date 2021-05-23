from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image
from gridizer import gridizer
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, help="an image's path")
parser.add_argument('--super', action ='store_true', default=False, help="turn on for leSuper")
args = parser.parse_args()

super = args.super
path = args.path

img = Image.open(requests.get(path, stream = True).raw)
imshow(img)
gridizer(img)

if super:
    pretrained_weights = torch.load('model.pt')
    model = models.resnet50(pretrained=True)
    model.load_state_dict(pretrained_weights)
else:
    img = cv.imread(path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(np.shape(img))
    cv.imwrite('sift_keys.jpg',img)
    kp, des = sift.detectAndCompute(gray,None)