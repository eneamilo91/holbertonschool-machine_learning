#!/usr/bin/env python3
"""
New code updates the script to take the last 10 columns of High and Close
and converts them into numpy.ndarray
"""


def array(df):
    """ performs manipulation """
    A = df.loc[:, ['High', 'Close']].tail(10).to_numpy()

    return A
