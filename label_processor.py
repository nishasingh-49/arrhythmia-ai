
import numpy as np
def binary_map(labels):
    """
    Converts labels into:
    0 -> Normal
    1 -> Abnormal
    """
    mapped = []
    for l in labels:
        if l == 'N':
            mapped.append(0)
        else:
            mapped.append(1)
    return np.array(mapped)
