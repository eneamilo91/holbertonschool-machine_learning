#!/usr/bin/env python3
'''
Module that calculates the cost of a neural network with L2 regularization
'''
import tensorflow as tf


def l2_reg_cost(cost, model):
    '''
    Calculates total cost including L2 regularization losses.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model containing regularized layers.
    base_cost : tf.Tensor
        The base loss (e.g., cross-entropy) without L2.

    Returns
    -------
    tf.Tensor
        Total cost including L2 regularization.
    '''
    reg_losses = tf.add_n(model.losses)
    return cost + reg_losses
