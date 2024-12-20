#!/usr/bin/env python3
"""New code updates Pandas DataFrame"""


def rename(df):
    """ renames columns """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df.loc[:, ['Datetime', 'Close']]

    return df
