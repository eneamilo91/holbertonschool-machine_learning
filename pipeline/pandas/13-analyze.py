#!/usr/bin/env python3
"""
Calculate descriptive statistics for all columns except Timestamp
"""

import pandas as pd


def analyze(df):
    """ computes descriptive stats """
    stats = df.drop(columns=['Timestamp']).describe()

    return (stats)
