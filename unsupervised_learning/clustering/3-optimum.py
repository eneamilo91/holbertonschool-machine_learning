#!/usr/bin/env python3
"""Optimum K method aka inversed elbow"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Optimun K method with variance"""
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(iterations, int) or iterations < 1:
            return None, None
        if kmax is not None and (type(kmax) is not int or kmax < 1):
            return None, None
        if kmax is not None and kmin >= kmax:
            return None, None
        if kmax is None:
            kmax = X.shape[0]
        if not isinstance(kmin, int) or kmin < 1 or kmin >= X.shape[0]:
            return None, None

        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            cluster, clss = kmeans(X, k, iterations)
            results.append((cluster, clss))
            variance_d = variance(X, cluster)
            if k == kmin:
                variance_k = variance_d
            d_vars.append(variance_k - variance_d)
        return results, d_vars
    except Exception:
        return None, None
