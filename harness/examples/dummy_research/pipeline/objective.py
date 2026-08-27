import numpy as np


def sample_weights(frame):
    return np.ones(len(frame.y), dtype=float)
