# AeroWF Downstream Task Contract V1

**File:** `DOWNSTREAM_TASK_CONTRACT_v1.md`  
**Companion machine-readable file:** `DOWNSTREAM_TASK_CONTRACT_v1.json`  
**Applies to:** AeroWF v1 downstream Forecast and Classification baselines  
**Normative status:** V1 task-layer protocol for model/evaluation alignment

> Important: the public AeroWF data schema defines the 21-class global weather vocabulary and states that `weather_label` and `weather_code_id` represent the same target. The **3-class collapse below is a project task-layer definition fixed by this contract**, not an upstream AeroWF vocabulary change.

---

## 1. Common time semantics

For one released sample with sample/window start time \(S\):

- runway input window length = 96 one-minute observations;
- runway input window covers `S ... S+95 min`;
- forecast origin \(O\) is the **last observation in the input window**, i.e. `O = S + 95 min`;
- released sample-start cadence is 15 minutes.

Therefore:

- `T+1` means `O + 1 min`;
- `T+4` means `O + 4 min`;
- `T+8` means `O + 8 min`.

**T+1/T+4/T+8 are one-minute forecast horizons after the input-window end. They are NOT +1/+4/+8 released rows, and NOT +15/+60/+120 minutes.**

All target lookup must be timestamp-based. Do not infer targets by blindly adding source row indices.

---

# 2. Forecast task

## 2.1 Forecast targets

Forecast the seven non-cyclic physical runway channels:

| Target index | Runway channel | Name |
|---:|---:|---|
| 0 | 0 | `cloud_base` |
| 1 | 1 | `wind_x` |
| 2 | 2 | `wind_y` |
| 3 | 3 | `pressure` |
| 4 | 4 | `temperature` |
| 5 | 5 | `humidity` |
| 6 | 6 | `dewpoint` |

The deterministic cyclic time channels `hour_sin`, `hour_cos`, `month_sin`, and `month_cos` are **inputs only** and are not forecast targets.

Padded runway slots (`runway_mask=False`) are never forecast/evaluated.

## 2.2 Exact T+1 / T+4 / T+8 target generation

For a sample starting at \(S\):

```text
input window:
    [S, S+1, ..., S+95] minutes

forecast origin:
    O = S+95

targets:
    T+1 = runway state at S+96
    T+4 = runway state at S+99
    T+8 = runway state at S+103
```

With the released 15-minute sample-start cadence, the canonical partner window starts at `S+15 min`. Its zero-based internal indices are:

| Horizon | Target timestamp | Partner-window internal index |
|---|---|---:|
| T+1 | `S+96 min` | 81 |
| T+4 | `S+99 min` | 84 |
| T+8 | `S+103 min` | 88 |

because:

```text
target_internal_index = (95 + horizon_minutes) - 15
                      = 80 + horizon_minutes
```

Normative construction:

1. locate the same-airport sample whose start timestamp is exactly `S+15 min`;
2. require that the partner sample belongs to the same released task partition as the anchor sample;
3. read channels `0..6` at internal indices `81`, `84`, `88`;
4. apply the anchor/airport `runway_mask`;
5. if any required partner or target point is unavailable, mark the anchor `target_available=false` and exclude it from that task/horizon;
6. never cross a train/validation/sealed task-partition boundary to create a target.

A target builder must report `n_anchor`, `n_target_available`, and `n_target_dropped` per airport, partition, and horizon.

## 2.3 Forecast input fields

Forecast models MAY use only information available in the anchor sample:

### Allowed

- `runway.npy` — all 11 input channels over the 96-minute history;
- `runway_mask.npy`;
- `exo_continuous.npy`:
  - `visibility`
  - `cloud_height`
  - `gust_speed`
- current-sample categorical exogenous variables:
  - `weather_code_id`
  - `sky_condition`
  - `has_gust`
  - `is_cavok`
- static graph fields:
  - `graph_edge_index.npy`
  - `graph_edge_type.npy`
- timestamp/cyclic time information from the anchor input only.

### Forbidden

- any future sample/window as model input;
- any forecast target value or direct derivative of a target;
- future exogenous fields;
- evaluation labels/results;
- split/status IDs, sample IDs, or source indices used as predictive features;
- `weather_label.npy` as an additional feature (it duplicates the weather-code target representation and is not needed when current `weather_code_id` is available).

---

# 3. Classification task

## 3.1 Source label

The source 21-ID weather vocabulary is:

| ID | Token |
|---:|---|
| 0 | `<PAD>` |
| 1 | `<UNK>` |
| 2 | `<GOOD_WX>` |
| 3 | `+TSRA` |
| 4 | `TSRA` |
| 5 | `TS` |
| 6 | `+SHRA` |
| 7 | `+RA` |
| 8 | `RASN` |
| 9 | `RA` |
| 10 | `PL` |
| 11 | `DZ` |
| 12 | `SN` |
| 13 | `SG` |
| 14 | `DS` |
| 15 | `FG` |
| 16 | `BR` |
| 17 | `HZ` |
| 18 | `DU` |
| 19 | `SA` |
| 20 | `SQ` |

`weather_label` and `weather_code_id` encode the same upstream weather-class target. Classification target generation must use the label only to create the target; it must not be exposed as an input.

## 3.2 Fixed 21-ID → 3-class task mapping

Task classes are:

```text
0 = GOOD
1 = PRECIP
2 = HAZARD
```

Mapping:

### Class 0 — GOOD

- ID 2 `<GOOD_WX>`

### Class 1 — PRECIP

- ID 6 `+SHRA`
- ID 7 `+RA`
- ID 8 `RASN`
- ID 9 `RA`
- ID 10 `PL`
- ID 11 `DZ`
- ID 12 `SN`
- ID 13 `SG`

### Class 2 — HAZARD

Thunderstorm / convective:

- ID 3 `+TSRA`
- ID 4 `TSRA`
- ID 5 `TS`

Visibility / dust / squall adverse weather:

- ID 14 `DS`
- ID 15 `FG`
- ID 16 `BR`
- ID 17 `HZ`
- ID 18 `DU`
- ID 19 `SA`
- ID 20 `SQ`

### Technical IDs

- ID 0 `<PAD>` → `IGNORE`
- ID 1 `<UNK>` → `IGNORE`

`<PAD>` and `<UNK>` are vocabulary-control tokens rather than meteorological classes. They must not be coerced into GOOD/PRECIP/HAZARD. Use `ignore_index=-100` for loss/evaluation and report the ignored sample count.

If the task owner later changes this collapse, that is a **task-contract version change** and must produce V2; do not silently alter the mapping inside baseline code.

## 3.3 Classification input fields

### Allowed

- `runway.npy` — all 11 runway-history channels;
- `runway_mask.npy`;
- `exo_continuous.npy`:
  - `visibility`
  - `cloud_height`
  - `gust_speed`
- categorical/binary context:
  - `sky_condition`
  - `has_gust`
  - `is_cavok`
- static graph fields;
- anchor timestamp/cyclic time information.

### Strictly forbidden

- `weather_code_id`;
- `weather_label`;
- one-hot/multi-hot encoding of `weather_code_id`;
- embedding lookup driven by `weather_code_id`;
- any severity flag, precipitation flag, thunderstorm flag, good-weather flag, or other deterministic/directly derived variable produced from `weather_code_id` or `weather_label`;
- any future information;
- evaluation labels/results;
- sample IDs, source indices, split/status fields as predictive features.

The classification exclusion applies both during training and evaluation.

---

# 4. Metrics

All headline metric values are reported as floating-point values, **not percentages**.

Examples:

```text
Macro-F1 = 0.731
CSI      = 0.614
```

not `73.1` or `61.4`.

## 4.1 Forecast MAE

For horizon \(h\) and target feature \(f\), over all valid `(sample, real-runway)` target points:

\[
MAE_{h,f} = \frac{1}{N_{h,f}}\sum_i |y_i-\hat y_i|
\]

## 4.2 Forecast RMSE

\[
RMSE_{h,f} =
\sqrt{
\frac{1}{N_{h,f}}
\sum_i (y_i-\hat y_i)^2
}
\]

### Forecast metric scale

**Primary baseline MAE/RMSE are computed on the released normalized runway scale `[0,1]`.**

Reasons:

- the model-side materialized runway tensors are the canonical frozen task inputs;
- the Min-Max transform is already frozen and shared;
- several upstream runway physical units are not safely inferable from the NumPy artifact alone;
- this avoids inconsistent unit conversion across baselines.

Reporting requirements:

- report MAE and RMSE separately for every horizon `{T+1,T+4,T+8}`;
- report separately for each of the seven target channels;
- padded runways and unavailable targets are excluded;
- if a single forecast summary is needed, use the unweighted arithmetic mean of the 21 feature-horizon values and label it explicitly:
  - `MAE_macro_norm`
  - `RMSE_macro_norm`.

Do not call the cross-feature average a physical-unit MAE/RMSE.

Optional source-scale metrics may be reported as secondary diagnostics only when both prediction and truth are transformed with the same canonical frozen statistics; they must be labeled `*_source_scale` and never replace the normalized primary metric.

## 4.3 Classification Macro-F1

For each evaluable class \(k\):

\[
Precision_k = \frac{TP_k}{TP_k+FP_k}
\]

\[
Recall_k = \frac{TP_k}{TP_k+FN_k}
\]

\[
F1_k =
\frac{2\,Precision_k\,Recall_k}
{Precision_k+Recall_k}
\]

Primary Macro-F1:

\[
MacroF1 = \frac{1}{|K_{eval}|}
\sum_{k\in K_{eval}}F1_k
\]

The three task labels must be passed explicitly as `[0,1,2]`; do not allow a library to infer class labels from predictions.

## 4.4 Classification CSI

For each evaluable class \(k\), one-vs-rest Critical Success Index:

\[
CSI_k = \frac{TP_k}{TP_k+FP_k+FN_k}
\]

True negatives are not included.

Required reporting:

- `CSI_GOOD`
- `CSI_PRECIP`
- `CSI_HAZARD`
- `CSI_macro`

The single field named `CSI` in a baseline summary means:

```text
CSI = CSI_macro
```

where `CSI_macro` is the unweighted mean over evaluable classes.

---

# 5. No-positive-class / zero-support policy

For class \(k\):

```text
support_k = TP_k + FN_k
```

If `support_k > 0`, compute F1 and CSI normally.

If `support_k == 0`:

- set `F1_k = NA`;
- set `CSI_k = NA`;
- exclude class \(k\) from the macro denominator for that report slice;
- still report `FP_k`;
- if `FP_k > 0`, additionally mark `false_positive_only=true`.

This rule prevents a class absent from ground truth from creating an arbitrary zero score while still exposing false alarms.

The evaluator must report:

```text
evaluated_classes
excluded_zero_support_classes
support_by_class
fp_by_class
```

If all three classes have zero support (which should not occur for a non-empty valid classification set), the metric is `NA` and the evaluation must fail validation rather than emit `0`.

Ignored `<PAD>/<UNK>` samples are removed before confusion-matrix construction and are counted separately as `n_ignored_labels`.

---

# 6. Aggregation and reporting

## Forecast

Minimum report matrix:

```text
3 horizons × 7 features × {MAE, RMSE}
```

plus optional normalized macro summaries.

Aggregation is over valid real-runway target points. Every real runway contributes one point; padded runway slots contribute nothing.

## Classification

Minimum report:

- sample count after ignore filtering;
- class support for GOOD/PRECIP/HAZARD;
- Macro-F1;
- per-class F1;
- CSI_GOOD / CSI_PRECIP / CSI_HAZARD;
- CSI_macro (`CSI`);
- zero-support exclusions, if any;
- confusion matrix ordered `[GOOD, PRECIP, HAZARD]`.

---

# 7. Leakage and split invariants

These rules apply to both tasks:

1. downstream train/validation/sealed boundaries from the frozen data release remain unchanged;
2. no target-generation operation may cross a released task-partition boundary;
3. normalization is never refitted using validation, temporal test, or ZBAD sealed data;
4. sealed temporal/spatial data are evaluation-only;
5. classification never receives `weather_code_id`, `weather_label`, or their direct deterministic derivatives as inputs;
6. target availability filtering must be deterministic and based only on timestamps/partition membership, never model output;
7. model and evaluator must use the same contract version.

---

# 8. Required baseline metadata

Every baseline result should record:

```text
task_contract_version = "1.0"
task = "forecast" | "classification"
data_release = "AeroWF_v1"
input_fields = [...]
target_definition = ...
metric_scale = ...
contract_sha256 = ...
```

The MD and JSON task contracts must travel together. The JSON file is the machine-readable source of truth for mapping, target offsets, allowed/forbidden fields, and metric configuration.
