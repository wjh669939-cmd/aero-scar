import numpy as np


def _masked_channel(frame, step, channel):
    values = frame.runway[:, :, step, channel]
    weights = frame.runway_mask.astype(np.float32)
    denom = np.maximum(weights.sum(axis=1), 1.0)
    return (values * weights).sum(axis=1) / denom


def featurize(frame):
    """MapLight-style compact baseline: frozen contract channels, mask-aware pool."""
    prev = np.asarray(frame.prev_wind_speed, dtype=float)
    temp = _masked_channel(frame, -2, 4)
    return np.column_stack(
        [
            prev,
            temp,
            np.asarray(frame.hour, dtype=float) / 24.0,
        ]
    )
