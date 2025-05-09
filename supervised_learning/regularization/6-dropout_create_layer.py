#!/usr/bin/env python3
'''Module that creates a Dense layer with dropout using TensorFlow 2.x'''
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''
    Creates a layer of a neural network using dropout.

    Parameters
    ----------
    prev : tf.Tensor
        Output from the previous layer.
    n : int
        Number of neurons in the layer.
    activation : callable
        Activation function to apply (e.g., tf.nn.relu).
    keep_prob : float
        Probability of keeping a node (i.e., 1 - dropout rate).

    Returns
    -------
    tf.Tensor
        Output of the new layer after applying dropout.
    '''
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    dense = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=init)(prev)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)(dense)
    return dropout
