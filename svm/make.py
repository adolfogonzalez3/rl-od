import numpy as np
from PIL import Image, ImageDraw
import os
path = os.getcwd()

data = []
for filename in os.listdir(path + "/stuff"):
    img = Image.open(path + "/stuff/" + filename)
    print(filename)
    img = img.resize((100, 100))
    img = np.array(img)[:, :, 0]
    data.append(img)
    print('lala')


data = np.array(data)
print(data.shape)
np.save("data", data)
exit()