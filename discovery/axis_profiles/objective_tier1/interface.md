# Interface contract (CONTRACT-TIER: changes require freeze process + coordinator sign-off)

- editable file: aerowf_downstream_v2/src/trial_objective.py ONLY;
- must keep signatures: forecast_loss(prediction, target, node_mask), classification_loss(logits, label, *, class_weights), compute_class_weights(train_label_counts);
- class statistics are computed from the train split by the locked caller; you may change how weights/losses are formed, not which split feeds them;
- losses must stay differentiable and finite on masked/virtual runways.
- TENSOR LAYOUT (factual, from the locked caller): prediction/target are (batch, runway_slots, horizons, components) = (B, 4 or 5, 3, 2); dim=1 is RUNWAY SLOTS, dim=2 is horizons [T+1, T+4, T+8], dim=3 is [wind_x, wind_y]; node_mask is (batch, runway_slots). Horizon-wise weights must be applied on dim=2, e.g. weights.view(1, 1, -1, 1).
