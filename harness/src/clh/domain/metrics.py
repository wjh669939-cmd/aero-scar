"""Independent metrics. Shared by dummy and AeroWF evaluators (paper eq. 3 inputs)."""

from __future__ import annotations

import numpy as np

from clh.core.errors import EvaluatorError
from clh.research.cards import MetricsBundle
from clh.research.reward import make_endpoints


def score_predictions(y_hat: np.ndarray, frame, *, split: str) -> MetricsBundle:
    y = np.asarray(frame.y, dtype=float)
    pred = np.asarray(y_hat, dtype=float)
    if pred.shape != y.shape:
        raise EvaluatorError(f"prediction shape {pred.shape} != target {y.shape}")
    if not np.isfinite(pred).all():
        raise EvaluatorError("predictions contain NaN/Inf")
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    hazard_pred = pred >= float(frame.hazard_threshold)
    hazard = np.asarray(frame.hazard, dtype=bool)
    tp = float(np.sum(hazard_pred & hazard))
    fp = float(np.sum(hazard_pred & ~hazard))
    fn = float(np.sum(~hazard_pred & hazard))
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    per_airport: dict[str, float] = {}
    for airport in sorted(set(map(str, frame.airports))):
        mask = np.asarray(frame.airports) == airport
        per_airport[airport] = float(np.mean(np.abs(pred[mask] - y[mask])))
    return MetricsBundle(
        mae=mae,
        rmse=rmse,
        hazard_csi=float(csi),
        per_airport_mae=per_airport,
        endpoints=make_endpoints(
            per_airport_mae=per_airport,
            hazard_csi=float(csi),
            overall_mae=mae,
        ),
        split=split,
        n_samples=int(y.shape[0]),
    )
