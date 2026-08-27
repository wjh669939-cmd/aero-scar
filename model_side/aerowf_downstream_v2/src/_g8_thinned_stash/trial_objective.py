"""O-tier1 trial surface: downstream losses and class-weight formula.

Train-split count collection and IGNORE_INDEX semantics stay in the locked
train scripts.  This file must not import trial_features or those scripts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def forecast_loss(
    prediction: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor
) -> torch.Tensor:
    mask = node_mask[:, :, None, None].expand_as(prediction)
    return torch.square(prediction - target)[mask].mean()


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
