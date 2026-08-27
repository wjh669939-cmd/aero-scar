# Parent seed3407 summary (val only, formal pipeline)

> Generated 2026-08-28 after . Full weights/results trees are NOT archived.
> Evaluator: C1 v1.0.2. . Checkpoint load missing/unexpected=0.

## Forecast (paper_table_aggregate macro over T+1/T+4/T+8)

| stage | RMSE_macro_norm | best_epoch |
|---|---|---|
| scratch | 0.04824 | 21 |
| pretrained | 0.05554 | 26 |

Per-horizon scratch RMSE_norm: T+1=0.04375, T+4=0.05068, T+8=0.05030.

## Classification

| stage | macro_f1 | CSI_macro | HAZARD f1 | best_epoch |
|---|---|---|---|---|
| scratch | 0.7307 | 0.6128 | 0.5877 | 7 |
| pretrained | 0.8137 | 0.7123 | 0.6341 | 29 |

## Cross-seed context (RMSE_macro scratch / pretrained)

| seed | fc_scr | fc_pre | cls_scr_f1 | cls_pre_f1 |
|---|---|---|---|---|
| 42 | 0.04854 | 0.05105 | 0.732 | 0.809 |
| 43 | 0.04812 | 0.05235 | 0.728 | 0.735 |
| 2027 | 0.04816 | 0.05268 | 0.710 | 0.745 |
| 3407 | 0.04824 | 0.05554 | 0.731 | 0.814 |

Note: seed5519 still running at archive time; will append in a later commit.
