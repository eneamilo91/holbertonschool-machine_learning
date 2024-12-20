#!/usr/bin/env python3
"""
Slice the DataFrame along the columns High, Low, Close, & Volume_BTC,
   taking every 60th row
"""


def slice(df):
    """ extracts columns """
    df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]

    return (df)
