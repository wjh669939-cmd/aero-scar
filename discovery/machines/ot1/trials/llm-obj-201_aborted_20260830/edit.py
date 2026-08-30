"""O-tier1 trial surface: downstream losses and class-weight formula.

Train-split count collection and IGNORE_INDEX semantics stay in the locked
train scripts.  This file must not import trial_features or those scripts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over active entries; returns zero for an empty active mask."""
    active = values[mask]
    count = max(active.numel(), 1)
    return active.sum() / count


def forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    event_mask: torch.Tensor | None = None,
    event_available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE plus masked vector cosine angular penalty.

    event_mask / event_available are accepted for compatibility with the O2
    event-weighted data precourse and are intentionally unused by this baseline
    patch.
    """
    mask = node_mask[:, :, None, None].expand_as(prediction)
    base_mse = _masked_mean(torch.square(prediction - target), mask)

    wind_x_pred = prediction[..., 0]
    wind_y_pred = prediction[..., 1]
    wind_x_targ = target[..., 0]
    wind_y_targ = target[..., 1]

    dot = wind_x_pred * wind_x_targ + wind_y_pred * wind_y_targ
    pred_norm = torch.sqrt(wind_x_pred * wind_x_pred + wind_y_pred * wind_y_pred)
    targ_norm = torch.sqrt(wind_x_targ * wind_x_targ + wind_y_targ * wind_y_targ)
    eps = 1e-6
    cos_sim = dot / (pred_norm * targ_norm + eps)

    angular_penalty = 1.0 - cos_sim
    active_mask = node_mask[:, :, None].expand_as(angular_penalty)
    angular_mean = _masked_mean(angular_penalty, active_mask)

    return base_mse + 0.2 * angular_mean


def compute_class_weights(train_label_counts: np.ndarray) -> torch.Tensor:
    counts = np.asarray(train_label_counts)
    n_classes = len(counts)
    weights = counts.sum() / (n_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def classification_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    *,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits, label, weight=class_weights, ignore_index=-100
    )
