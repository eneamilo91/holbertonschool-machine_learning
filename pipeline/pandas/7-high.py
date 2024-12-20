#!/usr/bin/env python3
"""
Sort the DataFrame by the High price in descending order
"""

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


def high(df):
    df = df.sort_values(by='High', ascending=False)

    return(df)
