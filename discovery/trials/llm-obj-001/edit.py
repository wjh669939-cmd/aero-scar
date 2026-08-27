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
    squared_error = torch.square(prediction - target)
    
    # Apply horizon-specific weights: [1, 1, 2] for horizons T+1, T+4, T+8
    # prediction shape: (batch, num_nodes, num_horizons, features)
    # The last dimension (index -1) is the horizon dimension
    num_horizons = prediction.shape[-1]
    weights = torch.ones(num_horizons, device=prediction.device, dtype=prediction.dtype)
    if num_horizons >= 3:
        weights[-1] = 2.0  # Weight the last horizon (T+8) by 2
    
    # Reshape weights to broadcast over batch, nodes, and features
    # weights shape: (1, 1, num_horizons, 1) to broadcast correctly
    weight_tensor = weights.view(1, 1, num_horizons, 1)
    
    weighted_squared_error = squared_error * weight_tensor
    return weighted_squared_error[mask].mean()


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
