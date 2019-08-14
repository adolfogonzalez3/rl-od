'''Test functions and classes in tf_utils.'''

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

import rl_od.common.tf_utils as tf_utils


def test_wrap_in_session():
    '''Test wrap_in_session dectorator.'''
    def _create_model():
        model = Sequential([InputLayer((5,)), Dense(1)])
        model.compile(optimizer='sgd', loss='binary_crossentropy')
        return model
    create_model = tf_utils.wrap_in_session(_create_model)
    model = create_model()
    assert isinstance(model, tf_utils.SessionWrapper)
    array = model.predict_on_batch(np.random.rand(10, 5))
    assert np.shape(array) == (10, 1)
    assert isinstance(model.metrics_names, list)


def test_call_in_session():
    '''Test call_in_session decorator.'''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.Graph()
    session = tf.Session(graph=graph, config=config)

    def _create_model():
        with graph.as_default(), session.as_default():
            model = Sequential([InputLayer((5,)), Dense(1)])
            model.compile(optimizer='sgd', loss='binary_crossentropy')
            return model
    model = _create_model()
    with pytest.raises(ValueError):
        model.predict_on_batch(np.random.rand(10, 5))
    predict_on_batch = tf_utils.call_in_session(
        model.predict_on_batch, session
    )
    array = predict_on_batch(np.random.rand(10, 5))
    assert np.shape(array) == (10, 1)


def test_create_conv_net():
    '''Test create_conv_net function.'''
    input_shape = (10, 10, 3)
    output_size = 10
    conv_net = tf_utils.create_conv_net(input_shape, output_size)
    array = conv_net.predict_on_batch(np.random.rand(10, *input_shape))
    assert np.shape(array) == (10, output_size)


def test_create_neural_net():
    '''Test create_neural_net function.'''
    input_shape = (10, 10, 3)
    output_size = 10
    nn_net = tf_utils.create_neural_net(input_shape, output_size)
    array = nn_net.predict_on_batch(np.random.rand(10, *input_shape))
    assert np.shape(array) == (10, output_size)
