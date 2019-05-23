from PIL import Image, ImageDraw
import random, os, time
import numpy as np

def randomImage():
    img = Image.open('image.png', 'r')
    img_w, img_h = img.size
    image = Image.new('RGB', (500, 500), (255, 255, 255, 255))
    bg_w, bg_h = image.size
    dx = dy = 100
    x = random.randint(0, bg_w-dx-1)
    y = random.randint(0, bg_h-dy-1)
    offset = ((x - img_w), (y - img_h))
    image.paste(img, (x, y))
    return image

def cropImage(image, agent_x, agent_y):
    # Crop is a fixed rectangle
    x_0 = agent_x + 23
    x_1 = agent_x - 23
    y_0 = agent_y + 23
    y_1 = agent_y - 23

    if (x_0, y_0) > image.size:
        return -1
    if x_1 < 0 or y_1 < 0:
        return -1

    area = (x_1, y_1, x_0, y_0)
    cropped_img = image.crop(area)
    return np.array(cropped_img)

def drawRect(image, agent_x, agent_y):
    x_0 = agent_x + 23
    x_1 = agent_x - 23
    y_0 = agent_y + 23
    y_1 = agent_y - 23

    draw = ImageDraw.Draw(image)
    draw.rectangle(((x_1, y_1), (x_0, y_0)), outline="Black")
    image.save("lala", "PNG")
