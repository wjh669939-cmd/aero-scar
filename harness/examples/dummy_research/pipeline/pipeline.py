"""Compose one trial. Evaluator calls fit_predict; this file is not axis-editable."""

from __future__ import annotations

from features import featurize
from model import Model
from objective import sample_weights
from physics import apply_physics


def fit_predict(train_frame, eval_frame):
    """Fit on evaluator-owned train (+ admitted extra data) and predict eval."""
    x_train = apply_physics(train_frame, featurize(train_frame))
    weights = sample_weights(train_frame)
    model = Model()
    model.fit(x_train, train_frame.y, sample_weight=weights)
    x_eval = apply_physics(eval_frame, featurize(eval_frame))
    return model.predict(x_eval)
