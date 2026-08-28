"""CPU functional smoke gate for axis objective_tier1 (ENGINEERING-TIER, owner-maintained).
Usage: python smoke_test.py <edited_file.py>
NOTE Phase A: the runner still uses its embedded copy; keep both in sync until
Phase B switches the runner to load this file. Every change must be logged in CHANGELOG.md.
"""

import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("trial_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# 关键：槽位数不得等于时距数（3）或分量数（2），否则维度猜测型 bug 逃逸
# （llm-obj-003 教训：真实训练 slots=4，与 horizon=3 不同形）
for n_slots in (5, 4):
    pred = torch.rand(4, n_slots, 3, 2); target = torch.rand(4, n_slots, 3, 2)
    node_mask = torch.tensor([[1] * (n_slots - 1) + [0]] * 4, dtype=torch.bool)
    loss = m.forecast_loss(pred, target, node_mask)
    assert loss.dim() == 0 and torch.isfinite(loss), f"forecast_loss bad (slots={n_slots}): {loss}"
    pred.requires_grad_(True)
    loss2 = m.forecast_loss(pred, target, node_mask); loss2.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
w = m.compute_class_weights(np.array([1000, 200, 50]))
assert w.shape == (3,) and torch.isfinite(w).all()
logits = torch.randn(8, 3, requires_grad=True)
label = torch.tensor([0, 1, 2, 0, 1, 2, 0, -100])
closs = m.classification_loss(logits, label, class_weights=w)
assert closs.dim() == 0 and torch.isfinite(closs); closs.backward()
print("SMOKE_OK")
