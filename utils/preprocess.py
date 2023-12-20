import numpy as np

def normalize(array, a, b):
    ratio = (array - np.min(array)) / (np.max(array) - np.min(array))
    return (b - a) * ratio + a