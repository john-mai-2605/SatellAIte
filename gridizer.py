# -*- coding: utf-8 -*-
"""Satell"AI"te.ipynb
by Hieu
"""

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image
import requests

url_day = "https://raw.githubusercontent.com/john-mai-2605/SatellAIte/main/data/test/daytime/day_7.png"
url_night = "https://raw.githubusercontent.com/john-mai-2605/SatellAIte/main/data/test/nighttime/day_7.png"

img = Image.open(requests.get(url_day, stream = True).raw)
imshow(img)

width, height = img.size

left = 100
top =  200
right = 1100
bottom = 800
cropsize = (left,top,right,bottom)

url_list = []
base_url = "https://raw.githubusercontent.com/john-mai-2605/SatellAIte/main/data/"
train_num = 6
test_num = 1
for idx in range(1,train_num+1):
    url_list.append(base_url + "train/daytime/day_" + str(idx) + ".png")
    url_list.append(base_url + "train/nighttime/night_" + str(idx) + ".png")
for idx in range(7,7+1):
    url_list.append(base_url + "test/daytime/day_" + str(idx) + ".png")
    url_list.append(base_url + "test/nighttime/night_" + str(idx) + ".png")

cropped_img_list = []
for url in url_list:
    try:
        tmp_img = Image.open(requests.get(url, stream=True).raw)
        tmp_img = tmp_img.crop(cropsize)
        cropped_img_list.append(tmp_img)
    except Exception as e:
        print(e)

from torchvision import transforms

pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
print(pil_to_tensor.shape) 

image_size = [600, 1000]

def gridizer(cropped_img):
    CELL_SIZE = 50
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    grid = []


    for c_left in range(GRID_WIDTH):
        for c_top in range(GRID_HEIGHT):
            tmp_cell = (c_left,c_top,c_left+CELL_SIZE,c_top+CELL_SIZE)
            tmp_img = cropped_img.crop(tmp_cell)
            grid.append(tmp_img)
    return grid

imshow(gridizer(cropped_img)[0])

