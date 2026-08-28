# Interface contract (CONTRACT-TIER: changes require freeze process + coordinator sign-off)

- editable file: aerowf_downstream_v2/src/trial_features.py ONLY;
- must keep signatures: build_forecast_inputs(...), AllowedContextEncoder(nn.Module), build_classification_inputs(...);
- forbidden input columns (asserted by the locked caller): weather_code_id, weather_label, significant_wx;
- target generation (what to predict) lives in locked files; you may only change how inputs are represented.
