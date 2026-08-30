"""O-tier1 trial surface: downstream losses and class-alpha formula.

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
    """Return inverse-frequency focal-loss alphas from training label counts.

    Classes absent from the train split receive zero alpha, avoiding infinite
    weights from zero counts while preserving the inverse-frequency scale for
    classes that are present.
    """
    counts = np.asarray(train_label_counts, dtype=np.float64)
    n_classes = len(counts)

    if n_classes == 0:
        return torch.empty(0, dtype=torch.float32)

    total = counts.sum()
    if total <= 0.0:
        return torch.zeros(n_classes, dtype=torch.float32)

    safe_counts = np.where(counts > 0.0, counts, 1.0)
    alpha = total / (n_classes * safe_counts)
    alpha = np.where(counts > 0.0, alpha, 0.0)
    return torch.tensor(alpha, dtype=torch.float32)


def classification_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    *,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Focal loss with inverse-frequency class alpha and gamma=2.0.

    Ignored positions (label == -100) are removed from the mean.  If all
    positions are ignored, returns a zero scalar that stays connected to
    logits so virtual-runway-only batches remain finite and differentiable.
    """
    gamma = 2.0
    ignore_index = -100

    label = label.to(device=logits.device)
    mask = label != ignore_index

    if not mask.any():
        return logits.sum() * 0.0

    label_clamped = label.masked_fill(~mask, 0)

    alpha = class_weights.to(device=logits.device, dtype=logits.dtype)
    alpha_t = alpha[label_clamped]

    ce_loss = F.cross_entropy(
        logits,
        label,
        weight=None,
        ignore_index=ignore_index,
        reduction="none",
    )

    log_probs = F.log_softmax(logits, dim=-1)
    log_p_t = log_probs.gather(dim=-1, index=label_clamped.unsqueeze(-1)).squeeze(-1)
    p_t = torch.exp(log_p_t)

    focal_per_element = (
        alpha_t
        * torch.pow((1.0 - p_t).clamp_min(0.0), gamma)
        * ce_loss
    )

    active_focal = focal_per_element[mask]
    return active_focal.sum() / active_focal.numel()
