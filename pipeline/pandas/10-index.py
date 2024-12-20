#!/usr/bin/env python3
"""
Index the DataFrame on the Timestamp column
"""


def index(df):
    """ sets index """
    df = df.set_index('Timestamp')

    return (df)
