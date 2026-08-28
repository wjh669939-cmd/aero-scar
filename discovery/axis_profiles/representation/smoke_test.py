"""CPU functional smoke gate for axis representation (ENGINEERING-TIER, owner-maintained).
Usage: python smoke_test.py <edited_file.py>
NOTE Phase A: the runner still uses its embedded copy; keep both in sync until
Phase B switches the runner to load this file. Every change must be logged in CHANGELOG.md.
"""

import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("trial_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# 槽位数用 4/5 两档，避开与时距(3)/分量(2)同形导致维度猜测型 bug 逃逸
for n_slots in (5, 4):
    runway = np.random.rand(96, n_slots, 2).astype(np.float32)
    mask = np.array([True] * (n_slots - 1) + [False])
    exo_cat = {"weather_code": 2, "sky_condition": 1, "has_gust": 0, "is_cavok": 1}
    exo_cont = np.array([0.5, 0.3, 0.0], dtype=np.float32)
    out = m.build_forecast_inputs(runway, mask, exo_cat, exo_cont, norm_stats=None)
    assert out["x"].shape == (96, n_slots, 2), f"forecast x shape {out['x'].shape}"
    assert out["node_mask"].dtype == torch.bool
    out2 = m.build_classification_inputs(runway, mask, exo_cat, exo_cont)
    assert out2["x"].shape == (96, n_slots, 2), f"cls x shape {out2['x'].shape}"
enc = m.AllowedContextEncoder(sky_known_max=5)
cat = {"sky_condition": torch.tensor([1, 2]), "has_gust": torch.tensor([0, 1]), "is_cavok": torch.tensor([1, 0])}
z = enc(cat, torch.rand(2, 3))
assert z.shape[0] == 2 and torch.isfinite(z).all()
print("SMOKE_OK")
