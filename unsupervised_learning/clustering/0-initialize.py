#!/usr/bin/env python3
"""
Defines function that initializes cluster centroids for K-means
"""


import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape

    low = np.min(X, axis=0)

    high = np.max(X, axis=0)

    centroids = np.random.uniform(low, high, size=(k, d))
    return centroids
