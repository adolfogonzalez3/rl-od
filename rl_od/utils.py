
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as k_layers
import tensorflow.keras.backend as k_backend


def create_convnet(input_shape=(80, 80, 3), n_classes=2):
    '''Create a convolutional neural network.'''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    k_backend.set_session(session)

    model = keras.Sequential()
    model.add(k_layers.InputLayer(input_shape))

    model.add(k_layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(k_layers.MaxPooling2D(2))
    model.add(k_layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(k_layers.MaxPooling2D(2))

    model.add(k_layers.Flatten())
    model.add(k_layers.Dense(256, activation='relu'))
    model.add(k_layers.Dropout(rate=0.3))
    model.add(k_layers.Dense(n_classes, activation='softmax'))

    return model

def crop_array(array, upper, left, bottom, right):
    '''
    Crop an array from using two coordinate points.
    
    :param array: (numpy.ndarray) An array with at least two dimensions.
    :param upper: (int) The upper bound of the crop.
    :param left: (int) The left bound of the crop.
    :param bottom: (int) The bottom bound of the crop.
    :param right: (int) The right bound of the crop.
    '''
    width = slice(bottom, upper, 1 if bottom < upper else -1)
    height = slice(left, right, 1 if left < right else -1)
    return array[width, height]

def compute_box_mask(shape, upper, left, bottom, right):
    '''
    Compute the bounding box using two coordinate points.
    
    :param array: (numpy.ndarray) An array with at least two dimensions.
    :param upper: (int) The upper bound of the crop.
    :param left: (int) The left bound of the crop.
    :param bottom: (int) The bottom bound of the crop.
    :param right: (int) The right bound of the crop.
    '''
    mask = np.zeros(shape)
    width = slice(bottom, upper, 1 if bottom < upper else -1)
    height = slice(left, right, 1 if left < right else -1)
    mask[width, height] = 1
    return mask

def draw_array(array, path):
    Image.fromarray(array).save(path)
    
