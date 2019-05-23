
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


