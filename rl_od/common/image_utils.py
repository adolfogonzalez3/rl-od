'''A module that contains functions and classes for manipulating images.'''
from collections import namedtuple

from PIL import Image

Coordinate = namedtuple('Coordinate', ['x', 'y'])
BoundingBox = namedtuple('BoundingBox', ['left', 'top', 'right', 'bottom'])


class ImageZoomer:
    '''A class for controlling the bounding box of an image.'''

    def __init__(self, image, bounding_box=None):
        self._image = image
        self.history = [BoundingBox(0, 0, *image.size)]
        if bounding_box:
            self.history.append(bounding_box)

    @classmethod
    def from_array(cls, array):
        '''
        Create an image from an array.

        :param array: (numpy.array) The array to convert into an image.
        :return: (ImageZoomer)
        '''
        return cls(Image.fromarray(array))

    @property
    def size(self):
        '''Return the size of the image.'''
        return self._image.size

    @property
    def current_bounding_box(self):
        '''Return the current bounding box.'''
        return self.history[-1]

    @property
    def image(self):
        '''Return the image with the appropiate bounding box.'''
        return self._image.resize(self.size, box=self.current_bounding_box)

    def pop(self):
        '''Pop off the last bounding box created.'''
        if len(self.history) > 1:
            self.history.pop()
        return self

    def adjust(self, width_pct, height_pct, translate_width, translate_height):
        '''
        Adjust the bounding box.

        :param width_pct: (float) Controls if the image is either expanded or
        shrunk width-wise.
        :param height_pct: (float) Controls if the image is either expanded or
        shrunk height-wise.
        :param translate_width: (float) Moves the bounding box in the image
        width-wise.
        :param translate_height: (float) Moves the bounding box in the image
        height-wise.
        '''
        left, top, right, bottom = self.current_bounding_box
        width = right - left
        height = bottom - top
        left = left + int(translate_width * width)
        top = top + int(translate_height * height)
        right = int(width * width_pct) + left
        bottom = int(height * height_pct) + top
        self.history.append(BoundingBox(left, top, right, bottom))
        return self.image

    def translate(self, translate_width, translate_height):
        '''
        Translate the bounding box.

        :param translate_width: (float) Moves the bounding box in the image
        width-wise.
        :param translate_height: (float) Moves the bounding box in the image
        height-wise.
        '''
        assert 0 <= translate_width <= 1
        assert 0 <= translate_height <= 1
        width_pct = 1 - translate_width
        height_pct = 1 - translate_height
        self.adjust(width_pct, height_pct, translate_width, translate_height)

    def shrink(self, width_pct, height_pct):
        '''
        Shrink the bounding box.

        :param width_pct: (float) Shrinks the image's width by a percentage of
        the current width.
        :param height_pct: (float) Shrinks the image's height by a percentage
        of the current height.
        '''
        assert 0 < width_pct <= 1
        assert 0 < height_pct <= 1
        self.adjust(width_pct, height_pct, 0, 0)
