AeroWF Data Contract V1 task for Closed-loop Auto Research.

Search-visible data: `release_v1/trainval/train` and `trainval/val` (plus pretrain as a data-axis extra).
Forbidden: `sealed/**`, `ZBAD`, `pretrain/test`.

The unedited pipeline is a compact persistence baseline on mask-pooled previous-step wind, analogous to the paper's strong CPU starting point. Model-axis Ridge should beat it; representation adds runway-relative wind from `wind_x`/`wind_y`.
