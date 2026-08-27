"""Named single-axis interventions used by the offline specialist and as LLM defaults."""

from __future__ import annotations

from pathlib import Path

from clh.config import AxisName
from clh.core.errors import AxisLockError
from clh.research.axis_lock import allowed_files

PRESETS: dict[tuple[AxisName, str], dict[str, str]] = {
    ("representation", "runway_wind"): {
        "features.py": '''import numpy as np

def featurize(frame):
    rad = np.deg2rad(frame.wind_dir - frame.runway_heading)
    headwind = frame.wind_speed * np.cos(rad)
    crosswind = frame.wind_speed * np.sin(rad)
    return np.column_stack([
        frame.wind_speed,
        frame.temp,
        frame.hour / 24.0,
        headwind,
        np.abs(crosswind),
    ])
'''
    },
    ("model", "ridge"): {
        "model.py": '''import numpy as np
from sklearn.linear_model import Ridge

class Model:
    name = "ridge"

    def __init__(self):
        self.est = Ridge(alpha=1.0)

    def fit(self, X, y, sample_weight=None):
        self.est.fit(np.asarray(X), np.asarray(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(np.asarray(X))
'''
    },
    ("physics", "extreme_wind_weights"): {
        "objective.py": '''import numpy as np

def sample_weights(frame):
    weights = np.ones(len(frame.y), dtype=float)
    weights[frame.wind_speed >= frame.hazard_threshold] = 5.0
    return weights
'''
    },
    ("objective", "extreme_wind_weights"): {
        "objective.py": '''import numpy as np

def sample_weights(frame):
    weights = np.ones(len(frame.y), dtype=float)
    weights[frame.wind_speed >= frame.hazard_threshold] = 5.0
    return weights
'''
    },
    ("data", "matched_airport"): {
        "data.py": '''def extra_source_ids():
    return ["matched_ZBHH"]
'''
    },
    ("data", "shifted_airport"): {
        "data.py": '''def extra_source_ids():
    return ["shifted_ZJHK"]
'''
    },
    ("data", "same_source_leak"): {
        "data.py": '''def extra_source_ids():
    return ["leak_ZBAA_future"]
'''
    },
}

AEROWF_PRESETS: dict[tuple[AxisName, str], dict[str, str]] = {
    ("representation", "runway_wind"): {
        "features.py": '''import numpy as np

def _masked_channel(frame, step, channel):
    values = frame.runway[:, :, step, channel]
    weights = frame.runway_mask.astype(np.float32)
    denom = np.maximum(weights.sum(axis=1), 1.0)
    return (values * weights).sum(axis=1) / denom

def featurize(frame):
    prev = np.asarray(frame.prev_wind_speed, dtype=float)
    wx = _masked_channel(frame, -2, 1)
    wy = _masked_channel(frame, -2, 2)
    temp = _masked_channel(frame, -2, 4)
    return np.column_stack([
        prev,
        temp,
        np.asarray(frame.hour, dtype=float) / 24.0,
        wx,
        wy,
        np.hypot(wx, wy),
    ])
'''
    },
    ("model", "ridge"): {
        "model.py": '''import numpy as np
from sklearn.linear_model import Ridge

class Model:
    name = "ridge"

    def __init__(self):
        self.est = Ridge(alpha=1.0)

    def fit(self, X, y, sample_weight=None):
        self.est.fit(np.asarray(X), np.asarray(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(np.asarray(X))
'''
    },
    ("physics", "extreme_wind_weights"): {
        "objective.py": '''import numpy as np

def sample_weights(frame):
    weights = np.ones(len(frame.y), dtype=float)
    weights[np.asarray(frame.y) >= float(frame.hazard_threshold)] = 5.0
    return weights
'''
    },
    ("objective", "extreme_wind_weights"): {
        "objective.py": '''import numpy as np

def sample_weights(frame):
    weights = np.ones(len(frame.y), dtype=float)
    weights[np.asarray(frame.y) >= float(frame.hazard_threshold)] = 5.0
    return weights
'''
    },
    ("data", "matched_airport"): {
        "data.py": '''def extra_source_ids():
    return ["pretrain_train"]
''',
        "external_manifest.json": '{"sources": ["pretrain_train"]}\n',
    },
    ("data", "shifted_airport"): {
        "data.py": '''def extra_source_ids():
    return ["shifted_climate"]
''',
        "external_manifest.json": '{"sources": ["shifted_climate"]}\n',
    },
    ("data", "same_source_leak"): {
        "data.py": '''def extra_source_ids():
    return ["leak_val"]
''',
        "external_manifest.json": '{"sources": ["leak_val"]}\n',
    },
}


def preset_files(
    axis: AxisName, name: str, pipeline_root: Path, *, domain: str = "dummy"
) -> dict[str, str]:
    if not name:
        return {}
    table = AEROWF_PRESETS if domain.lower() in {"aerowf", "atc"} else PRESETS
    key = (axis, name)
    if key not in table:
        raise AxisLockError(f"unknown preset {name!r} for axis {axis}")
    files = dict(table[key])
    extra = allowed_files(axis) - set(files)
    for rel in extra:
        src = pipeline_root / rel
        if src.exists():
            files[rel] = src.read_text(encoding="utf-8")
    return files
