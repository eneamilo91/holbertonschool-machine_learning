#!/usr/bin/env python3
"""
New code fills in the missing data points in the DataFrame
"""


def fill(df):
    """ performs actions """
    df = df.drop(columns=['Weighted_Price'])

    df['Close'].fillna(method='pad', inplace=True)

    df['High'].fillna(df.Close, inplace=True)
    df['Low'].fillna(df.Close, inplace=True)
    df['Open'].fillna(df.Close, inplace=True)

    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)

    return (df)
