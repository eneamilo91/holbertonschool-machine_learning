#!/usr/bin/env python3
"""
Rearrange the MultiIndex levels such that the timestamp is first
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """ concatenates dataframes """
    df1 = df1.loc[
        (df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)]
    df2 = df2.loc[
        (df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)]

    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    df = df.reorder_levels([1, 0], axis=0)

    df = df.sort_index()

    print(df)
