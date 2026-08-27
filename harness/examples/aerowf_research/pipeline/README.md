Editable AeroWF pipeline for Closed-loop Auto Research (arXiv:2606.22731).

Axis lock (one trial, one axis):

| Axis | Editable files |
| --- | --- |
| representation | `features.py` |
| model | `model.py` |
| physics | `physics.py`, `objective.py` |
| data | `data.py`, `external_manifest.json` |

`pipeline.py` is evaluator-owned glue. Search scores only `trainval/val`.
Held-out labels live in evaluator-owned sealed / temporal holdout splits.
Tensors follow `DATA_CONTRACT_v1.md`. Do not refit `train_stats_v1.json`.
PRE2020 `weather_label.npy` must not be used as an SSL target.
