# Parent seed5519 summary (val only, formal pipeline)

> Generated 2026-08-28 after `PARENT_5519_EXIT=0` / `PARENT_NIGHT_DONE`.
> Full weights/results trees are NOT archived. Evaluator: C1 v1.0.2. `test_used=false`.
> checkpoint_load missing/unexpected: 0/0 on both pretrained legs. Pipeline elapsed ~9935 s (~2.76 h).

## Forecast (paper_table_aggregate macro over T+1/T+4/T+8)

| stage | RMSE_macro_norm | best_epoch |
|---|---|---|
| scratch | 0.04820 | 20 |
| pretrained | 0.05203 | 29 |

Per-horizon scratch RMSE_norm: T+1=0.04554, T+4=0.04887, T+8=0.05019.

## Classification

| stage | macro_f1 | HAZARD f1 | best_epoch |
|---|---|---|---|
| scratch | 0.7564 | 0.6806 | 3 |
| pretrained | 0.8024 | 0.7089 | 21 |

## Five-seed parent table (RMSE_macro / macro_f1)

| seed | fc_scr | fc_pre | delta(pre-scr) | cls_scr | cls_pre | delta_f1 |
|---|---|---|---|---|---|---|
| 42 | 0.04854 | 0.05105 | +0.00252 | 0.732 | 0.809 | +0.077 |
| 43 | 0.04812 | 0.05235 | +0.00423 | 0.728 | 0.735 | +0.007 |
| 2027 | 0.04816 | 0.05268 | +0.00451 | 0.710 | 0.745 | +0.035 |
| 3407 | 0.04824 | 0.05554 | +0.00730 | 0.731 | 0.814 | +0.083 |
| 5519 | 0.04820 | 0.05203 | +0.00383 | 0.756 | 0.802 | +0.046 |

Notes: forecast scratch cluster 0.0481-0.0485 across all five seeds; pretraining negative
transfer on forecast is same-sign on all five; classification pretrained gain same-sign on all five.
