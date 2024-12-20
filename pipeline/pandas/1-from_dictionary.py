#!/usr/bin/env python3
"""
Creates a Pandas DataFrame from a dictionary and saves it into variable df
"""


import pandas as pd


def from_dictionary():
    """Creates a Pandas DataFrame from a dictionary"""
    df = pd.DataFrame(
        {
            "First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]
        },
        index=list("ABCD"))
    return df


df = from_dictionary()
