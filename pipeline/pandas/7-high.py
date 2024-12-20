#!/usr/bin/env python3
"""
Sort the DataFrame by the High price in descending order
"""


def high(df):
    """ sorts by high price """
    df = df.sort_values(by='High', ascending=False)

    return (df)
