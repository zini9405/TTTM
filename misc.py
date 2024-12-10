import os
import numpy as np
import pandas as pd


def mode(
    arr: list
) -> bool:
    
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    arr = arr[~pd.isna(arr)]

    if len(arr) == 0:
        return None

    unique, counts = np.unique(arr, return_counts = True)
    index = np.argmax(counts)
    return unique[index]


def get_fname(
    path: str
) -> str:
    
    fname = os.path.split(path)[-1]
    fname, ext = os.path.splitext(fname)
    return fname
