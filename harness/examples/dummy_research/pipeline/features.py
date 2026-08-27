import numpy as np


def featurize(frame):
    return np.column_stack(
        [
            frame.wind_speed,
            frame.temp,
            frame.hour / 24.0,
        ]
    )
