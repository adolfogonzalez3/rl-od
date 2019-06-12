
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence


class FaceDataSequence(Sequence):
    '''A class used to get images from the face data set.'''

    def __init__(self, batch_size=32, train=True):
        package_path = Path(__file__).parent
        if train is True:
            data_path = package_path / 'FaceData' / 'RL'
        else:
            data_path = package_path / 'FaceData' / 'Disc'
        self.image_paths = [im_path
                            for im_path in (data_path / 'Disc').iterdir()
                            if im_path.is_file()]
        self.batch_size = batch_size
        self.train = train

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, idx):
        begin = idx*self.batch_size
        end = begin + self.batch_size
        idx = slice(begin, end if end <= len(self) else None)
        img = np.array([np.array(Image.open(im_path))
                        for im_path in self.image_paths[idx]],
                        dtype=np.float) / 255
        return im, np.full((self.batch_size,), int(self.train))


def iter_imgs(path):
    return (Image.open(im) for im in path.iterdir() if im.is_file())

def convert_imgs_to_grayscale(iterator):
    return (im.convert('L') for im in iterator)

def resize_imgs(iterator, new_shape):
    return (im.resize(new_shape, Image.ANTIALIAS) for im in iterator)

def convert_to_array(iterator):
    return (np.array(it) for it in iterator)

def load_data():
    package_path = Path(__file__).parent
    data_path = package_path / 'FaceData'

    test_img = iter_imgs(data_path / 'Disc')
    test_img = convert_imgs_to_grayscale(test_img)
    test_img = np.array(list(convert_to_array(test_img)))
    test_img = np.expand_dims(test_img, axis=-1)
    #test_img = [np.asarray(Image.open(im_path).convert('L'))
    #            for im_path in (data_path / 'Disc').iterdir()
    #            if im_path.is_file()]
    #tr_img = [np.array(Image.open(im_path).resize((300, 225), Image.ANTIALIAS))
    #          for im_path in (data_path / 'RL').iterdir()
    #          if im_path.is_file()]
    #tr_img = [np.array(Image.open(im_path).resize((80, 80), Image.ANTIALIAS).convert('L'))
    #          for im_path in (data_path / 'RL').iterdir()
    #          if im_path.is_file()]
    tr_img = iter_imgs(data_path / 'RL')
    tr_img = convert_imgs_to_grayscale(tr_img)
    tr_img = resize_imgs(tr_img, (80, 80))
    tr_img = np.array(list(convert_to_array(tr_img)))
    tr_img = np.expand_dims(tr_img, axis=-1)

    return np.asarray(tr_img)[:800], np.asarray(test_img)[:200]


if __name__ == '__main__':
    train, test = load_data()
    print(train.shape)
