#!/usr/bin/env python3
"""
Index the DataFrame on the Timestamp columns and concatenates them
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """ concatenates dataframes """
    df2 = df2.loc[df2['Timestamp'] <= 1417411920]

    df1 = df1.index('Timestamp')
    df2 = df2.index('Timestamp')

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return (df)
