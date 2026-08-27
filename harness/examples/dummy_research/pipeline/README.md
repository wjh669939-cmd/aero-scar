Editable AeroWF-shaped pipeline for Closed-loop Auto Research.

Axis lock (one trial, one axis):

| Axis | Editable files |
| --- | --- |
| representation | `features.py` |
| model | `model.py` |
| physics | `physics.py`, `objective.py` |
| data | `data.py`, `external_manifest.json` |

`pipeline.py` is evaluator-owned glue. The search loop never receives hidden-split labels.
