"""O-tier1 trial surface: downstream losses and class-weight formula.

Train-split count collection and IGNORE_INDEX semantics stay in the locked
train scripts.  This file must not import trial_features or those scripts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    event_mask: torch.Tensor | None = None,
    event_available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked component MSE plus a wind-speed magnitude penalty.

    The component MSE is computed exactly as the baseline.  A differentiable
    speed term ``sqrt(ux^2 + uy^2 + 1e-8)`` couples the two Cartesian wind
    components and penalizes errors in wind-speed magnitude with lambda = 1.0.
    ``event_mask``/``event_available`` are accepted for caller compatibility
    but are intentionally unused by this baseline-style forecast loss.
    """
    del event_mask, event_available

    eps = 1e-8

    # Component MSE, masked per runway slot and expanded over horizons/components.
    component_mask = node_mask[:, :, None, None].expand_as(prediction)
    component_loss = torch.square(prediction - target)[component_mask].mean()

    # Wind-speed magnitude, computed per element from Cartesian components.
    pred_speed = torch.sqrt(
        torch.square(prediction[..., 0])
        + torch.square(prediction[..., 1])
        + eps
    )
    target_speed = torch.sqrt(
        torch.square(target[..., 0])
        + torch.square(target[..., 1])
        + eps
    )

    speed_mask = node_mask[:, :, None].expand_as(pred_speed)
    speed_loss = torch.square(pred_speed - target_speed)[speed_mask].mean()

    lambda_speed = 1.0
    return component_loss + lambda_speed * speed_loss


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
