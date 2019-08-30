'''Tests the image_utils.py module.'''

import numpy as np
import numpy.random as npr

import rl_od.common.image_utils as image_utils


def create_checkerboard():
    '''Return an array with a checkboard pattern.'''
    image = np.empty((16, 16))
    image[8:, :8] = image[:8, 8:] = 2
    image[8:, 8:] = 4
    image = (image * 64 - 1).astype(np.uint8)
    image[:8, :8] = 0
    return image


def test_ImageZoomer_from_array():
    '''Test ImageZoomer from_array class method.'''
    image_array = npr.randint(255, size=(16, 16))
    image_zoomer = image_utils.ImageZoomer.from_array(image_array)
    assert image_zoomer.size == (16, 16)
    assert np.all(np.array(image_zoomer.image) == image_array)


def test_ImageZoomer_adjust():
    '''Test ImageZoomer adjust method.'''
    image_zoomer = image_utils.ImageZoomer.from_array(create_checkerboard())
    image_zoomer.adjust(0.5, 0.5, 0, 0)
    assert np.all(np.array(image_zoomer.image) == 0)
    image_zoomer.pop()
    image_zoomer.adjust(0.5, 0.5, 0.5, 0)
    assert np.all(np.array(image_zoomer.image) == 127)
    image_zoomer.pop()
    image_zoomer.adjust(0.5, 0.5, 0, 0.5)
    assert np.all(np.array(image_zoomer.image) == 127)
    image_zoomer.pop()
    image_zoomer.adjust(0.5, 0.5, 0.5, 0.5)
    assert np.all(np.array(image_zoomer.image) == 255)


def test_ImageZoomer_shrink():
    '''Test ImageZoomer shrink method.'''
    image_zoomer = image_utils.ImageZoomer.from_array(create_checkerboard())
    image_zoomer.shrink(0.5, 0.5)
    assert np.all(np.array(image_zoomer.image) == 0)
    image_zoomer.pop()
    image_zoomer.shrink(0.75, 0.75)
    assert not np.all(np.array(image_zoomer.image) == 0)


def test_ImageZoomer_translate():
    '''Test ImageZoomer translate method.'''
    image_zoomer = image_utils.ImageZoomer.from_array(create_checkerboard())
    image_zoomer.translate(0.5, 0)
    assert set(np.array(image_zoomer.image).ravel()) == {127, 255}
    image_zoomer.pop()
    image_zoomer.translate(0, 0.5)
    assert set(np.array(image_zoomer.image).ravel()) == {127, 255}
    image_zoomer.pop()
    image_zoomer.translate(0.5, 0.5)
    assert np.all(np.array(image_zoomer.image) == 255)
