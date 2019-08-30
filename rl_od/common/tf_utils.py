'''A module that contains classes and functions for using tensorflow.'''

from contextlib import contextmanager

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten,
                                     InputLayer)


def wrap_in_session(function, session=None):
    '''
    Wraps an object returned by a function in a SessionWrapper.

    :param function: (callable) A callable that returns an object.
    :param session: (None or tensorflow.Session) A session that is wrapped
    over all methods and attributes of the returned object.
    :return: (SessionWrapper) A session wrapped object.
    '''
    def _wrapped_function(*args, **kwargs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        graph = session.graph if session else tf.Graph()
        with graph.as_default():
            new_session = session or tf.Session(graph=graph, config=config)
            with new_session.as_default():
                returned_object = function(*args, **kwargs)
        return SessionWrapper(returned_object, new_session)
    return _wrapped_function


def call_in_session(function, session):
    '''
    Wraps a function call in a session scope.

    :param function: (callable) A callable to call within a session scope.
    :param session: (None or tensorflow.Session) A session that is scoped
    over the function call.
    :return: () Returns the result of function.
    '''
    def _wrapped_function(*args, **kwargs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        graph = session.graph
        with graph.as_default():
            with session.as_default():
                return function(*args, **kwargs)
    return _wrapped_function


@wrap_in_session
def create_conv_net(input_shape, output_size, kernel_sizes=(3, 3),
                    filter_sizes=(32, 64), layers=(256, 256),
                    activation='relu'):
    '''
    Create a wrapped keras convnet with its own private session.

    :param input_shape: (Sequence) The shape of the expected input.
    :param output_size: (int) The number of labels intended to be predicted.
    :param kernel_sizes: (Sequence) Defines the sizes of the kernels.
    :param filter_sizes: (Sequence) Defines the number of filters.
    :param layers: (Sequence) Defines the number of hidden layers.
    :param activations: (str) Defines the activation function to use.
    :return: (WrappedSession(tf.keras.Model)) A keras model
    '''
    model = Sequential()
    model.add(InputLayer(input_shape))
    for k_size, f_size in zip(kernel_sizes, filter_sizes):
        model.add(Conv2D(
            f_size, kernel_size=k_size, activation=activation, padding='same'
        ))
        model.add(MaxPooling2D(2))
    model.add(Flatten())
    for hidden_units in layers:
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(
        optimizer="adam", loss='binary_crossentropy', metrics=['accuracy']
    )
    return model


@wrap_in_session
def create_neural_net(input_shape, output_size, layers=(256, 256),
                      activation='relu'):
    '''
    Create a wrapped keras neural network with its own private session.

    :param input_shape: (Sequence) The shape of the expected input.
    :param output_size: (int) The number of labels intended to be predicted.
    :param layers: (Sequence) Defines the number of hidden layers.
    :param activations: (str) Defines the activation function to use.
    :return: (WrappedSession(tf.keras.Model)) A keras model
    '''
    model = Sequential()
    model.add(InputLayer(input_shape))
    if len(input_shape) > 1:
        model.add(Flatten())
    for hidden_units in layers:
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(
        optimizer="adam", loss='binary_crossentropy', metrics=['accuracy']
    )
    return model


class SessionWrapper:
    '''A class that encapsulates all methods of a class in a session.'''

    def __init__(self, model, session):
        '''
        Create a session wrapper.

        :param model: () An object that will have all its methods wrapped with
        a session.
        :param session: (tensorflow.Session) Used to wrap all method calls.
        '''
        self._wrapped_model = model
        self._session = session

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        with self.with_scope():
            returned_attr = getattr(self._wrapped_model, attr)
            if callable(returned_attr):
                return call_in_session(returned_attr, self._session)
            return returned_attr

    def __repr__(self):
        return '<SessionWrapper<{!r}>>'.format(self._wrapped_model)

    @contextmanager
    def with_scope(self):
        '''Enter into the owned session's scope.'''
        with self._session.as_default(), self._session.graph.as_default():
            yield
