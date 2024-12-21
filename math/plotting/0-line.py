#!/usr/bin/env python3
""" module of a function to plot a line graph"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """ function that plots a line graph """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r-')
    plt.xlim((0, 10))
    plt.show()
