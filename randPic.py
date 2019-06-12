from PIL import Image, ImageDraw
import random, os, time
import numpy as np
from os.path import isfile, join
import os

def getDiscImage(path, file):
    img = Image.open(path + "Disc/" + file, 'r')
    return np.array(img)

def getRLImage(path, file):
    img = Image.open(path + "RL/" + file, 'r')
    img.thumbnail((300, 300), Image.ANTIALIAS)
    imgsave = img
    convert = img
    return np.array(convert), np.array(imgsave)

def showImage(image):
    img = Image.fromarray(image, "RGB")
    img.show()

def cropImage(image, agent_x, agent_y):
    # Crop is a fixed rectangle
    x0 = agent_x + 40
    x1 = agent_x - 40
    y0 = agent_y + 40
    y1 = agent_y - 40

    if x0 >= image.shape[0] or y0 >= image.shape[1]:
        return -1
    if x1 < 0 or y1 < 0:
        return -1
    area = image[x1:x0, y1:y0].copy()
    return area

def drawRect(image, agent_x, agent_y, ind, flag, goal):
    #img = Image.fromarray(image, "RGB")
    img = Image.fromarray(np.uint8((image)*255))
    x_0 = agent_x + 40
    x_1 = agent_x - 40
    y_0 = agent_y + 40
    y_1 = agent_y - 40

    if ind % 400 == 0 or flag and goal == 0:
        img.save("./pics1/%d %d.jpg" % (ind, goal), "JPEG")
    return np.array(img)
    
def drawRect(image, agent_x, agent_y, ind, flag, goal):
    #img = Image.fromarray(image, "RGB")
    img = Image.fromarray(np.uint8((image)*255))
    x_0 = agent_x + 40
    x_1 = agent_x - 40
    y_0 = agent_y + 40
    y_1 = agent_y - 40

    if ind % 400 == 0 or flag and goal == 0:
        img.save("./pics1/%d %d.jpg" % (ind, goal))
    return np.array(img)

def cropper(image, x0, x1, y0, y1):

    if x0 > 225 - 80:
        x0 = 225 - 80
    if y0 > 300 - 80:
        y0 = 300 - 80

    if x0 >= x1:
        x1 = x0 + 80
    if y0 >= y1:
        y1 = y0 + 80

    if x1 - x0 < 80:
        x1 += 80
    if y1 - y0 < 80:
        y1 += 80
    area = image[x0:x1, y0:y1].copy()
    '''
    print("x0", x0)
    print("x1", x1)
    print("y0", y0)
    print("y1", y1)
    print(area.shape)
    '''
    img = Image.fromarray(np.uint8((area)*255))
    re = img.resize((80, 80), Image.ANTIALIAS)
    return area, np.array(re) / 255

def paster(image):
    img = Image.fromarray(np.uint8((image)*255))
    img_w, img_h = img.size
    background = Image.new('RGBA', (225, 300), (255, 255, 255, 255))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    return np.array(background) / 255
    
