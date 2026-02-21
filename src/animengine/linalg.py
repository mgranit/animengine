import numpy as np
def project(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # dot and norm squared along the last axis, keep dims for broadcasting
    dot = np.sum(a * b, axis=-1, keepdims=True)
    norm_sq = np.sum(b * b, axis=-1, keepdims=True)
    # avoid division by zero
    scale = np.where(norm_sq == 0, 0.0, dot / norm_sq)
    return scale * b

def reflect(a, b):
    a = np.asarray(a, dtype=float)
    return 2 * project(a, b) - a