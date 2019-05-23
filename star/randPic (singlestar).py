from PIL import Image, ImageDraw
import random, os, time
import numpy as np

def randomImage():
    img = Image.open('image.png', 'r')
    img_w, img_h = img.size
    size = 150
    image = Image.new('RGB', (size, size), (255, 255, 255, 255))
    bg_w, bg_h = image.size
    dx = dy = 44
    x = random.randint(0, bg_w-dx-1)
    y = random.randint(0, bg_h-dy-1)
    offset = ((x - img_w), (y - img_h))
    image.paste(img, (x, y))
    image.save("curr", "PNG")
    return image

def randFaceImage():
    img = Image.open('image.png', 'r')
    img_w, img_h = img.size
    size = 150
    image = Image.new('RGB', (size, size), (255, 255, 255, 255))
    bg_w, bg_h = image.size
    dx = dy = 44
    x = random.randint(0, bg_w-dx-1)
    y = random.randint(0, bg_h-dy-1)
    offset = ((x - img_w), (y - img_h))
    image.paste(img, (x, y))
    image.save("curr", "PNG")
    return image


def cropImage(image, agent_x, agent_y):
    # Crop is a fixed rectangle
    x_0 = agent_x + 40
    x_1 = agent_x - 40
    y_0 = agent_y + 40
    y_1 = agent_y - 40

    if (x_0, y_0) > image.size:
        return -1
    if x_1 < 0 or y_1 < 0:
        return -1

    area = (x_1, y_1, x_0, y_0)
    cropped_img = image.crop(area)
    return np.array(cropped_img.convert("1"))

def drawRect(image, agent_x, agent_y, ind, flag):
    image = Image.open('curr', 'r')
    x_0 = agent_x + 40
    x_1 = agent_x - 40
    y_0 = agent_y + 40
    y_1 = agent_y - 40

    draw = ImageDraw.Draw(image)
    draw.rectangle(((x_0, y_0), (x_1, y_1)), outline="Black")
    if ind % 400 == 0 or flag:
        image.save("./pics/%d" % ind, "PNG")
    return np.array(image)
    
    
