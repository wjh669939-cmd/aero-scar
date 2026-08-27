# DOWNSTREAM_TASK_CONTRACT V1 → V2 Change Notice

## Effective version

`DOWNSTREAM_TASK_CONTRACT_v2` is now the active task-layer contract.

## Breaking Forecast change

V1 formal Forecast:
- `T+1/T+4/T+8` = 1/4/8 minutes after input end
- 7 physical runway targets

V2 formal Forecast:
- `T+1` = 15 minutes
- `T+4` = 60 minutes
- `T+8` = 120 minutes
- targets = `wind_x`, `wind_y`

The former V1 minute-level seven-variable task is retained only as:
- `H+1m`
- `H+4m`
- `H+8m`

and must be reported as an optional extension.

## Classification

No substantive task change:
- 21-ID → GOOD/PRECIP/HAZARD mapping retained
- `<PAD>/<UNK>` ignored
- `weather_code_id`, `weather_label`, and direct deterministic derivatives forbidden as classification inputs
- Macro-F1 / CSI / zero-support rules retained

## Required action

Model side:
- stop using V1 Forecast semantics for the formal baseline
- use V2 for new runs
- rerun any formal Forecast baseline produced under V1 minute-level semantics

Evaluation side:
- evaluate formal Forecast at 15/60/120 minutes on `wind_x/wind_y`
- reject formal-baseline results that still use V1 minute-level semantics

Keep V1 files for audit history; use V2 as the current contract.
