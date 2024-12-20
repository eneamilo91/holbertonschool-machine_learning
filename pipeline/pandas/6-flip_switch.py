#!/usr/bin/env python3
"""
Transpose the rows and columns and then sorts the data
    in reverse chronological order
"""


def flip_switch(df):
    """ sorts the data and transposes """
    df = df.sort_values(by='Timestamp', ascending=False).T

    return (df)
