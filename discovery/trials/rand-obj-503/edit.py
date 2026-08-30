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
    """Baseline masked MSE.

    event_mask / event_available: (batch, horizons) bool，仅训练批提供（O2 事件加权
    的数据前置，2026-08-29 通管）。基线实现不使用它们——行为与旧签名完全一致；
    事件加权类动作可消费，且必须尊重 event_available（不可用时距不得当普通负样本）。
    """
    mask = node_mask[:, :, None, None].expand_as(prediction)
    weights = torch.tensor([3, 1, 1], device=prediction.device, dtype=prediction.dtype)
    weighted = torch.square(prediction - target) * weights.view(1, 1, -1, 1)
    return weighted[mask].mean()


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
