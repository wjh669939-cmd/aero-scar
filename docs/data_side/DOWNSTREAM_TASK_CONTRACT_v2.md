# AeroWF Downstream Task Contract V2

**Applies to:** AeroWF v1 downstream Forecast and Classification baselines  
**Status:** CURRENT / ACTIVE

> `DOWNSTREAM_TASK_CONTRACT_v1` is superseded for **Forecast** semantics.  
> Classification is carried forward unchanged in substance.  
> Model-side and evaluation-side code should use **V2** from this point forward.

## 1. Formal Forecast baseline

The formal/paper-aligned Forecast baseline is:

- `T+1` = 15 minutes after input-window end
- `T+4` = 60 minutes after input-window end
- `T+8` = 120 minutes after input-window end
- targets = runway-level `wind_x`, `wind_y`

For anchor window start `S`:

```text
input window = S ... S+95 min
forecast origin O = S+95 min

T+1 target time = S+110 min
T+4 target time = S+155 min
T+8 target time = S+215 min
```

Because released sample starts are 15 minutes apart:

| Horizon | Partner window start | Target internal index |
|---|---:|---:|
| T+1 | `S+15 min` | 95 |
| T+4 | `S+60 min` | 95 |
| T+8 | `S+120 min` | 95 |

Implementations must align by timestamp, not blindly by row number. The partner must be the same airport and the same released task partition. Never cross train/val/sealed boundaries to create a target.

Only `runway_mask=True` runway slots are valid for loss/evaluation.

### Forecast target channels

| Target index | Runway channel | Feature |
|---:|---:|---|
| 0 | 1 | `wind_x` |
| 1 | 2 | `wind_y` |

### Allowed Forecast inputs

- all 11 historical runway channels
- `runway_mask`
- `exo_continuous`: `visibility`, `cloud_height`, `gust_speed`
- current categorical context: `weather_code_id`, `sky_condition`, `has_gust`, `is_cavok`
- graph structure
- anchor timestamp / cyclic time

### Forbidden Forecast inputs

- future windows
- future exogenous fields
- target values or target-derived fields
- evaluation labels/results
- split/status/sample ID/source index as predictive features

## 2. Optional extended Forecast

The old V1 minute-level task is retained only as an optional extension and must be renamed:

- `H+1m`
- `H+4m`
- `H+8m`

Targets:

- `cloud_base`
- `wind_x`
- `wind_y`
- `pressure`
- `temperature`
- `humidity`
- `dewpoint`

Do not call this optional task `T+1/T+4/T+8`, and do not mix its results with the formal baseline table.

## 3. Classification

Classes:

```text
0 = GOOD
1 = PRECIP
2 = HAZARD
```

### 21-ID → 3-class mapping

GOOD:
- 2 `<GOOD_WX>`

PRECIP:
- 6 `+SHRA`
- 7 `+RA`
- 8 `RASN`
- 9 `RA`
- 10 `PL`
- 11 `DZ`
- 12 `SN`
- 13 `SG`

HAZARD:
- 3 `+TSRA`
- 4 `TSRA`
- 5 `TS`
- 14 `DS`
- 15 `FG`
- 16 `BR`
- 17 `HZ`
- 18 `DU`
- 19 `SA`
- 20 `SQ`

IGNORE:
- 0 `<PAD>`
- 1 `<UNK>`

Use `ignore_index=-100`.

### Classification allowed inputs

- all 11 historical runway channels
- `runway_mask`
- `exo_continuous`: `visibility`, `cloud_height`, `gust_speed`
- `sky_condition`
- `has_gust`
- `is_cavok`
- graph fields
- anchor timestamp / cyclic time

### Classification strictly forbidden inputs

- `weather_code_id`
- `weather_label`
- one-hot/multi-hot encoding of either
- embeddings indexed by `weather_code_id`
- any direct deterministic derivative of `weather_code_id` or `weather_label`
- future information
- evaluation labels/results
- sample ID/source index/split/status as predictive features

## 4. Metrics

### Forecast MAE

For horizon `h` and component `f ∈ {wind_x, wind_y}`:

`MAE = mean(abs(y_true - y_pred))`

### Forecast RMSE

`RMSE = sqrt(mean((y_true - y_pred)^2))`

Compute over valid sample × real-runway points only.

Primary project metric scale: released normalized runway scale `[0,1]`.

Required formal Forecast report:

```text
3 horizons × 2 wind components × {MAE, RMSE}
```

Optional summaries:
- `MAE_macro_norm`
- `RMSE_macro_norm`

### Classification Macro-F1

Compute per-class F1 for explicit labels `[0,1,2]`, then unweighted mean across evaluable classes.

### CSI

Per class:

`CSI_k = TP_k / (TP_k + FP_k + FN_k)`

Required:
- `CSI_GOOD`
- `CSI_PRECIP`
- `CSI_HAZARD`
- `CSI_macro`

If a single field is named `CSI`, it means `CSI_macro`.

## 5. Zero-support policy

For class `k`:

`support_k = TP_k + FN_k`

If `support_k > 0`:
- compute F1 and CSI normally.

If `support_k == 0`:
- `F1_k = NA`
- `CSI_k = NA`
- exclude that class from the macro denominator for that report slice
- still report `FP_k`
- if `FP_k > 0`, report `false_positive_only=true`

If all three classes have zero support, evaluation is invalid and the metric is `NA`.

Ignored `<PAD>/<UNK>` samples are removed before confusion-matrix construction and counted separately.

## 6. Versioning

Formal baseline result metadata must record:

```text
task_contract_version = "2.0"
data_release = "AeroWF_v1"
task = "forecast" | "classification"
forecast_protocol = "paper_aligned_wind_forecast" |
                    "extended_minute_multivariate_forecast" |
                    null
metric_scale = ...
contract_sha256 = ...
```

V1 files should be retained for audit traceability, but V2 is the active contract.
