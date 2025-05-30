#!/usr/bin/env python3
"""
    Specificity
"""

import numpy as np


def specificity(confusion):
    """
        Function to calculates specificity of each class
        in a confusion matrix

        :param confusion: ndarray, shape(classes,classes) confusion matrix

        :return: ndarray, shape(classes,), specificity of each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize specificity
    specificity_matrix = np.zeros((classes,))

    # formula of specificity : true negative / (true negative + false positive)
    for i in range(classes):
        true_pos = confusion[i, i]
        false_pos = np.sum(confusion[:, i]) - true_pos
        false_neg = np.sum(confusion[i, :]) - true_pos
        true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)
        specificity_matrix[i] = true_neg / (true_neg + false_pos)

    return specificity_matrix
