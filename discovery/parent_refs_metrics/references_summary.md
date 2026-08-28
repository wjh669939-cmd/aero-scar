# Reference baselines (D3, frozen evaluator v1.0.2, val only)

> Generated 2026-08-28. Same contract as formal trials: full-coverage export
> (11883 rows) -> predictions_adapter -> C1 evaluator subprocess. No training
> for persistence/majority. Scripts: `aerowf_downstream_v2/src/reference_persistence_export.py`
> (persistence); majority-class built from frozen val label alignment.

## Forecast — Persistence (prediction = last observed frame, all horizons)

| metric | value | parent scratch (5-seed) |
|---|---|---|
| RMSE_macro_norm | **0.073148** | 0.04825 ± 0.00017 |
| MAE_macro_norm | **0.024479** | 0.02545 (seed42) |

Notes: parent beats persistence by ~34% on the primary RMSE metric.
Persistence MAE is slightly *better* than parent — the classic persistence
signature (many tiny errors, catastrophic misses on regime changes); RMSE
punishes the misses, which is why the primary metric is RMSE_macro_norm.

## Classification — Majority class (always GOOD)

| metric | value | parent scratch (seed42) |
|---|---|---|
| macro_f1 | **0.318661** | 0.75035 |
| CSI_macro | **0.305226** | 0.62679 |
| hazard_class_f1 | **0.000000** | 0.63309 |

## Still pending (D3)

- Logistic-regression reference (feature choice to be fixed with experiment
  side; CPU-only, can run alongside discovery batches).
