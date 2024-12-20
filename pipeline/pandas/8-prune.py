#!/usr/bin/env python3
"""
New code removes the entries in the DataFrame where Close is NaN
"""


def prune(df):
    """ removes entries """
    df = df.dropna(subset=['Close'])

    return (df)
